from app import app, server


def test_app_initialization():
    assert app is not None
    assert callable(server)
