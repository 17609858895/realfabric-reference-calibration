"""Validate the release structure without importing or training analysis models."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_ZIP_SHA256='bc65ccde56b153c45ce8c8fc39e9ec129c9a83c2a1df4841ee95fc944d8681f6'
def main():
    scripts=sorted((ROOT/'scripts').glob('*.py'))
    for path in scripts:ast.parse(path.read_text(encoding='utf-8'))
    z=ROOT/'dataset_for_eg2024.zip'
    report={'python_modules_parsed':len(scripts),'training_executed':False,'source_archive_present':z.is_file()}
    if z.is_file():
        h=hashlib.sha256()
        with z.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
        report['archive_sha256']=h.hexdigest()
        assert h.hexdigest()==EXPECTED_ZIP_SHA256,'Source archive differs from the analysed release.'
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
