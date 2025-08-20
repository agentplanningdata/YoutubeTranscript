"""
requirements.txt 파일 테스트

프로젝트의 requirements.txt 파일이 존재하고 필수 패키지들이 포함되어 있는지 확인합니다.
"""

from pathlib import Path


def test_requirements_file_exists():
    """requirements.txt 파일이 존재하는지 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    assert requirements_file.exists(), "requirements.txt 파일이 존재하지 않습니다."
    assert requirements_file.is_file(), "requirements.txt가 파일이 아닙니다."


def test_requirements_contains_basic_packages():
    """requirements.txt에 기본 필수 패키지들이 포함되어 있는지 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    # 파일이 존재한다고 가정하고 내용 확인
    content = requirements_file.read_text(encoding="utf-8").lower()
    
    # README.md에서 명시된 기본 패키지들
    required_packages = [
        "fastapi",
        "gradio", 
        "langgraph",
        "chromadb",
        "psycopg2-binary",  # PostgreSQL driver
        "youtube_transcript_api",
        "google-api-python-client",  # YouTube Data API
        "openai",
        "langchain",
        "pytest",
        "python-dotenv"
    ]
    
    for package in required_packages:
        assert package in content, f"필수 패키지 '{package}'가 requirements.txt에 없습니다."


def test_requirements_has_proper_format():
    """requirements.txt 파일이 올바른 형식으로 작성되어 있는지 확인"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    content = requirements_file.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
    
    # 최소한 몇 개의 패키지는 있어야 함
    assert len(lines) >= 5, "requirements.txt에 너무 적은 패키지가 있습니다."
    
    # 각 줄이 패키지 형식인지 확인 (간단한 검증)
    for line in lines:
        # 빈 줄이나 주석이 아닌 경우, 패키지 이름이 있어야 함
        if line and not line.startswith("#"):
            # 패키지 이름은 문자로 시작해야 함
            assert line[0].isalpha() or line[0] in ['-', '_'], f"잘못된 패키지 형식: {line}" 