"""J7: copy the P0 input identities computed by phase-1 track MAP into a standalone manifest (no file is moved or re-hashed)."""
import json, hashlib, sys
src = 'results/audit97/MAP/dependency_map.json'
d = json.load(open(src, encoding='utf-8'))
raw = open(src, 'rb').read()
out = {
    'manifest': 'WL97 P0 input identity manifest (track J7)',
    'source': src,
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'note': 'sha256_16 = first 16 hex digits of sha256 as computed by phase-1 MAP on LabServer63 (user nd). J7 did not re-hash; files were not moved. matches_record = MAP compared the prefix with the prefix recorded in the cited worklog/result.',
    'server': 'LabServer63, user nd',
    'artifacts': d['artifact_identities'],
    'protocol_hashes': d.get('protocol_hashes'),
}
json.dump(out, open('results/audit97/J7/p0_input_manifest.json', 'w', encoding='utf-8', newline='\n'), indent=1)
a = d['artifact_identities']
print(len(a), 'artifacts;', sum(1 for x in a if x.get('matches_record') is True), 'match record;', [x['id'] for x in a if x.get('matches_record') is not True])
