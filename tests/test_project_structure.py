"""
프로젝트 기본 구조 테스트

YouTube Transcript PoC 프로젝트의 기본 디렉토리 구조가 올바르게 설정되었는지 확인합니다.
"""

import os
from pathlib import Path


def test_project_root_directories_exist():
    """프로젝트 루트의 기본 디렉토리들이 존재하는지 확인"""
    project_root = Path(__file__).parent.parent
    
    required_directories = [
        "src",
        "tests", 
        "docs",
        "scripts"
    ]
    
    for directory in required_directories:
        dir_path = project_root / directory
        assert dir_path.exists(), f"필수 디렉토리 '{directory}'가 존재하지 않습니다."
        assert dir_path.is_dir(), f"'{directory}'가 디렉토리가 아닙니다."


def test_src_subdirectories_exist():
    """src 디렉토리 내부의 주요 하위 디렉토리들이 존재하는지 확인"""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    required_subdirectories = [
        "agents",
        "api", 
        "core",
        "database",
        "services",
        "utils",
        "ui"
    ]
    
    for subdirectory in required_subdirectories:
        subdir_path = src_path / subdirectory
        assert subdir_path.exists(), f"src/{subdirectory} 디렉토리가 존재하지 않습니다."
        assert subdir_path.is_dir(), f"src/{subdirectory}가 디렉토리가 아닙니다."


def test_init_files_exist():
    """Python 패키지를 위한 __init__.py 파일들이 존재하는지 확인"""
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    # src 디렉토리 자체에 __init__.py
    assert (src_path / "__init__.py").exists(), "src/__init__.py 파일이 존재하지 않습니다."
    
    # 주요 하위 디렉토리들에 __init__.py  
    subdirectories = ["agents", "api", "core", "database", "services", "utils", "ui"]
    
    for subdirectory in subdirectories:
        init_file = src_path / subdirectory / "__init__.py"
        assert init_file.exists(), f"src/{subdirectory}/__init__.py 파일이 존재하지 않습니다."


def test_project_has_proper_structure():
    """전체 프로젝트 구조가 올바른지 종합 확인"""
    project_root = Path(__file__).parent.parent
    
    # 프로젝트 루트에 있어야 할 파일들
    required_files = [
        "README.md",
        "plan.md"
    ]
    
    for file_name in required_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"프로젝트 루트에 '{file_name}' 파일이 존재하지 않습니다."
        assert file_path.is_file(), f"'{file_name}'이 파일이 아닙니다." 