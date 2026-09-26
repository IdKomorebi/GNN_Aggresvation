"""Create a flat Elsevier source archive; run from any directory. No data needed."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
root = Path(__file__).resolve().parents[1]
files = sorted(root.glob('*.tex')) + sorted((root/'tables').glob('*.tex'))
files += [root/'ref.bib'] + sorted((root/'figures').glob('*.pdf'))
if (root/'main.bbl').is_file(): files.append(root/'main.bbl')
assert len({p.name for p in files}) == len(files), 'Flat archive filename collision'
with ZipFile(root/'submission_sources.zip', 'w', ZIP_DEFLATED) as z:
    for p in files:
        if p.suffix == '.tex':
            txt = p.read_text().replace(r'\graphicspath{{figures/}}', r'\graphicspath{{./}}')
            txt = re.sub(r'(\\input\{)tables/', r'\1', txt)
            z.writestr(p.name, txt)
        else:
            z.write(p, p.name)
print('Created', root/'submission_sources.zip', 'with', len(files), 'flat files')
