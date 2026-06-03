from news_retrieval_agent.main import main


def test_main(capsys) -> None:
    main()
    captured = capsys.readouterr()
    assert "Hello from NewsRetrievalAgent!" in captured.out
