"""Advisory pre-commit checks; NOT a complete secret scanner. Run before git add."""
from pathlib import Path
import re, sys
root = Path(__file__).resolve().parents[1]
forbidden_roots = {'checkpoints','models','predictions','artifacts','lime_html'}
forbidden_ext = {'.npy','.npz','.pt','.pth','.bin','.safetensors','.pkl','.pickle','.joblib','.h5','.ckpt'}
private_names = {'kaggle.json','credentials.json','errors_to_tag.csv','T14_lime_explanations.csv'}
patterns = {
 'GitHub PAT': re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),
 'HuggingFace token': re.compile(r'hf_[A-Za-z0-9]{25,}'),
 'OpenAI key': re.compile(r'sk-[A-Za-z0-9_-]{20,}'),
 'Email-shaped string': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}
issues=[]
scanned=0
for path in root.rglob('*'):
 if not path.is_file() or '.git' in path.parts: continue
 rel=path.relative_to(root)
 if any(part in forbidden_roots for part in rel.parts) or path.suffix.lower() in forbidden_ext or path.name in private_names:
  issues.append(f'Private/large artifact present: {rel}')
 if path.stat().st_size>40*1024*1024:issues.append(f'Large file >40 MiB: {rel}')
 if path.suffix.lower() in ('.py','.md','.json','.ipynb','.txt','.csv') and path.stat().st_size<10*1024*1024:
  scanned += 1
  data=path.read_text(encoding='utf-8',errors='ignore')
  for name,rex in patterns.items():
   if rex.search(data): issues.append(f'{name} — manual review needed in {rel}')
print(f'Checked {scanned} readable files.')
if issues:
 print('WARNING / MANUAL REVIEW:')
 for i in sorted(set(issues)): print(' -',i)
 sys.exit(1)
print('No obvious forbidden artifacts or token patterns. Still review all notebooks before publishing.')
