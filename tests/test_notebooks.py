"""Ensure examples are clean notebooks whose Python cells can be compiled."""
from pathlib import Path
import json
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


@pytest.mark.parametrize('name', [
    'spabound_mouse_embryo.ipynb',
    'spabound_hln_a1.ipynb',
    'spabound_hln_d1.ipynb',
])
def test_clean_executable_cells(name):
    notebook = json.loads((EXAMPLES / name).read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    assert notebook['nbformat_minor'] >= 5
    ids = [cell['id'] for cell in notebook['cells']]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(notebook['cells']):
        assert cell['cell_type'] in ('code', 'markdown', 'raw')
        assert isinstance(cell['metadata'], dict)
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None
            assert not cell['outputs']
            compile(source, f'{name}:cell{index}', 'exec')
            assert 'spatialVAE' not in source
            assert 'SpatialGlue_MMVAE' not in source
