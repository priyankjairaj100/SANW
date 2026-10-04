#!/usr/bin/env python3
"""Repair an incomplete local TeX configuration without changing ACL sources.

Uses installed TeX Live resources and URW base-35 Type 1 fonts. No system files
are modified. Normal Overleaf and complete TeX installations do not need this.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess


def configure() -> dict[str, str]:
    source = pathlib.Path(__file__).resolve().parent
    runtime = source / 'tex-runtime'
    tree = runtime / 'texmf'
    for name in ('formats', 'var', 'config'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    font_dir = tree / 'fonts/type1/urw'
    font_dir.mkdir(parents=True, exist_ok=True)
    map_dir = tree / 'fonts/map/pdftex'
    map_dir.mkdir(parents=True, exist_ok=True)
    names = {
        'Times-Roman': 'NimbusRoman-Regular', 'Times-Italic': 'NimbusRoman-Italic',
        'Times-Bold': 'NimbusRoman-Bold', 'Times-BoldItalic': 'NimbusRoman-BoldItalic',
        'Helvetica': 'NimbusSans-Regular', 'Helvetica-Oblique': 'NimbusSans-Italic',
        'Helvetica-Bold': 'NimbusSans-Bold', 'Helvetica-BoldOblique': 'NimbusSans-BoldItalic',
        'Helvetica-Narrow': 'NimbusSansNarrow-Regular',
        'Helvetica-Narrow-Oblique': 'NimbusSansNarrow-Oblique',
        'Helvetica-Narrow-Bold': 'NimbusSansNarrow-Bold',
        'Helvetica-Narrow-BoldOblique': 'NimbusSansNarrow-BoldOblique',
        'Courier': 'NimbusMonoPS-Regular', 'Courier-Oblique': 'NimbusMonoPS-Italic',
        'Courier-Bold': 'NimbusMonoPS-Bold', 'Courier-BoldOblique': 'NimbusMonoPS-BoldItalic',
    }
    tex = pathlib.Path('/usr/share/texlive/texmf-dist')
    lines = []
    for line in (tex / 'fonts/map/dvips/psnfss/psnfss.map').read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] not in names:
            continue
        original = fields[1]
        replacement = names[original]
        font_source = pathlib.Path('/usr/share/fonts/X11/Type1') / (replacement + '.pfb')
        if not font_source.is_file():
            raise RuntimeError(f'Missing installed scalable font: {font_source}')
        shutil.copy2(font_source, font_dir / font_source.name)
        lines.append(line.replace(original, replacement, 1).rstrip() + ' <' + font_source.name)
    for map_name in ('cm', 'cmextra', 'symbols', 'latxfont', 'euler'):
        lines.extend((tex / 'fonts/map/dvips/amsfonts' / (map_name + '.map')).read_text().splitlines())
    lines.extend(pathlib.Path('/usr/share/texmf/fonts/map/dvips/lm/lm.map').read_text().splitlines())
    font_map = map_dir / 'pdftex.map'
    font_map.write_text('\n'.join(lines) + '\n')
    env = os.environ.copy()
    env.update({'TEXMF': '{' + str(tree) + ',/usr/share/texmf,/usr/share/texlive/texmf-dist}',
                'TEXMFVAR': str(runtime / 'var'), 'TEXMFCONFIG': str(runtime / 'config'),
                'TEXFORMATS': str(runtime / 'formats') + ':'})
    fmt = runtime / 'formats/pdflatex.fmt'
    if not fmt.exists():
        cmd = ['pdftex', '-ini', '-etex', '-jobname=pdflatex', '-progname=pdflatex',
               '-output-directory=' + str(runtime / 'formats'), '\\input pdflatex.ini']
        run = subprocess.run(cmd, env=env, cwd=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (runtime / 'format-build.log').write_text(run.stdout)
        if run.returncode:
            raise RuntimeError('Local TeX format generation failed; inspect tex-runtime/format-build.log')
    receipt = {'map_sha256': hashlib.sha256(font_map.read_bytes()).hexdigest(),
               'mapped_base35_fonts': len([line for line in lines if line.startswith(('ptm', 'pcr', 'phv'))]),
               'font_files': [p.name for p in sorted(font_dir.glob('*.pfb'))],
               'purpose': 'Local font-map repair; official ACL style and document metrics unchanged.'}
    (source / 'local_tex_repair_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return env


if __name__ == '__main__':
    configured = configure()
    print(json.dumps({key: configured[key] for key in ('TEXMF', 'TEXMFVAR', 'TEXMFCONFIG', 'TEXFORMATS')}, indent=2))
