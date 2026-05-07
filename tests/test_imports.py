def test_imports():
    import waveslab
    from waveslab.covers import row_mean_cover

    assert waveslab is not None
    assert callable(row_mean_cover)
