from mask_fit_feat.io import load_trial


def test_load_trial():
    data = load_trial('data/rsc_blaine-fit_20250522_144305.csv')
    assert len(data['Pa_Global']) > 1000
    assert data['Pa_Global'].index.freq.delta.total_seconds() == 0.001
    assert data['mask_particles'].index.freq.delta.total_seconds() == 1.0
