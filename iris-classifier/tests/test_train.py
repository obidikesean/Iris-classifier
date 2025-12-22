from train import train_model

def test_train_model():
    """Tests the train_model function."""
    # Run training without saving files
    accuracy = train_model(test_size=0.2, random_state=42, save_outputs=False)
    # Assert that accuracy is 1.0 for this specific random state
    assert accuracy == 1.0
