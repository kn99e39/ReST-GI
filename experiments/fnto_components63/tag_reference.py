"""Observational event tags; original action, path equations and total sink stay intact."""
import ast,copy,hashlib,inspect,weakref
import torch

def instrument_walker(FW):
    src=inspect.getsource(FW.walk);original=ast.parse(src);tree=copy.deepcopy(original)
    class AddTag(ast.NodeTransformer):
        count=0
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call):
                f=node.value.func
                if isinstance(f,ast.Attribute) and f.attr=='index_add_' and isinstance(f.value,ast.Attribute) and f.value.attr=='T' and isinstance(f.value.value,ast.Name) and f.value.value.id=='acc':
                    extra=ast.parse('acc._wl63_zero_collision = (step == 1)').body[0];extra._wl63_added=True;self.count+=1
                    return [extra,node]
            return self.generic_visit(node)
    add=AddTag();tree=add.visit(tree);assert add.count==1,'Only the existing first-exit sink call may be tagged'
    class Strip(ast.NodeTransformer):
        def visit_Assign(self,node):return None if getattr(node,'_wl63_added',False) else self.generic_visit(node)
    stripped=Strip().visit(copy.deepcopy(tree))
    assert ast.dump(stripped,include_attributes=False)==ast.dump(original,include_attributes=False),'Path/RNG/weight AST changed'
    ast.fix_missing_locations(tree);namespace=dict(FW.__dict__)
    exec(compile(tree,'<wl63-observational-first-exit-tag>','exec'),namespace)
    proof=dict(original_source_sha256=hashlib.sha256(src.encode()).hexdigest(),original_AST_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),tagged_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),strip_added_assignment_restores_original_AST=True,added_assignments=add.count,added_RNG_calls=0)
    return namespace['walk'],proof

def tagged_accumulator(AO,created):
    Original=AO.Acc
    class Sink:
        def __init__(self,owner,untagged):self.owner=weakref.ref(owner);self.untagged=untagged
        def index_add_(self,dim,idx,we):
            a=self.owner();assert type(a._wl63_zero_collision) is bool
            # Untagged original total and moments are computed by ORIGINAL code on the SAME events.
            self.untagged.index_add_(dim,idx,we)
            row=idx//a.nsrc;col=idx%a.nsrc;name='B' if a._wl63_zero_collision else 'Scat'
            a.event_counts[name]+=len(row)
            for k in range(len(a.fields)):
                weighted=we*a.ratio[k,col].T
                for target in ('T',name):
                    a.tag_first[target][k].index_add_(1,row,weighted)
                    a.tag_second[target][k].index_add_(1,row,weighted.square())
    class TaggedAcc(Original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.tag_first={k:torch.zeros_like(self.values) for k in ('T','B','Scat')}
            self.tag_second={k:torch.zeros_like(self.second) for k in ('T','B','Scat')}
            self.event_counts=dict(B=0,Scat=0);self._wl63_zero_collision=None
            self.T=Sink(self,self.T);created.append(self)
    return TaggedAcc

def relative_l1(a,b):return float((a-b).abs().sum()/b.abs().sum().clamp(min=1e-300))

def tagged_action(AO,FW,dom,field,N,seed):
    created=[];TaggedAcc=tagged_accumulator(AO,created);walk,proof=instrument_walker(FW)
    old_acc,old_walk=AO.Acc,FW.walk
    try:
        AO.Acc,FW.walk=TaggedAcc,walk
        result=AO.action(dom,field,N,seed)
    finally:AO.Acc,FW.walk=old_acc,old_walk
    assert len(created)==1;a=created[0]
    shape=(len(a.fields),dom.P,64,3)
    unpack=lambda y:y.permute(0,2,1).reshape(shape).cpu()
    first={k:unpack(v) for k,v in a.tag_first.items()};second={k:unpack(v) for k,v in a.tag_second.items()}
    original=unpack(a.values);original_second=unpack(a.second)
    gates=dict(original_vs_tagged_total=relative_l1(first['T'],original),original_vs_tagged_second=relative_l1(second['T'],original_second),component_first_closure=relative_l1(first['B']+first['Scat'],first['T']),component_second_closure=relative_l1(second['B']+second['Scat'],second['T']),untagged_float32_response_matches=bool(torch.equal(result['response'],original.float())),exclusive_exit_count_matches=bool(sum(a.event_counts.values())==a.stats['exits']))
    result.update(component_first=first,component_second=second,raw_BS_cross_moment=0.,raw_BS_cross_reason='Each photon exits once, in exactly one mutually exclusive zero/collided event class; not zero statistical covariance',event_counts=a.event_counts,instrumentation=proof,gates_A_B=gates)
    return result
