from app.cli import main


def test_cli_one_shot(capsys):
    exit_code = main([
        "--prompt", "Add the chopped onion",
        "--num-suggestions", "2",
        "--length", "4",
        "--strategy", "greedy",
    ])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Loaded" in output
    assert "1." in output
    assert "2." in output
