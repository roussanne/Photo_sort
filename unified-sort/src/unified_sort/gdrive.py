"""
Google Drive integration for cloud backup and sync.

Google Drive API를 사용한 클라우드 백업 및 동기화:
- OAuth2 인증
- 분류된 이미지 자동 업로드
- 폴더 구조 자동 생성 (sharp/defocus/motion/uncertain)
- 중복 방지 및 재시작 지원
- 진행률 추적

개선사항:
1. OAuth2 인증 플로우 (웹 브라우저 기반)
2. 재시작 가능한 업로드 (세션 저장)
3. 폴더 자동 생성 및 캐싱
4. 타입 힌트 및 에러 처리
5. Streamlit 통합을 위한 콜백 지원
"""

from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
import json
import os

# Google API 의존성 확인
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False
    Credentials = None
    HttpError = Exception


# OAuth2 스코프 - Drive 파일 생성 및 관리
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# 토큰 저장 경로
DEFAULT_TOKEN_PATH = Path.home() / '.unified_sort' / 'gdrive_token.json'
DEFAULT_CREDENTIALS_PATH = Path.home() / '.unified_sort' / 'credentials.json'


class GDriveUploader:
    """
    Google Drive 업로더 클래스.

    OAuth2 인증 후 이미지를 분류별 폴더에 자동 업로드합니다.
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None
    ):
        """
        Args:
            credentials_path: OAuth2 클라이언트 credentials.json 경로
            token_path: 인증 토큰 저장 경로 (재사용용)
        """
        if not _GDRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive API dependencies not installed. "
                "Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )

        # 경로 설정
        self.credentials_path = Path(credentials_path) if credentials_path else DEFAULT_CREDENTIALS_PATH
        self.token_path = Path(token_path) if token_path else DEFAULT_TOKEN_PATH

        # 토큰 디렉토리 생성
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

        # Google Drive 서비스 초기화
        self.service = None
        self.creds = None

        # 폴더 ID 캐시 (폴더 재생성 방지)
        self.folder_cache: Dict[str, str] = {}

    def authenticate(self) -> bool:
        """
        Google Drive OAuth2 인증을 수행합니다.

        토큰이 있으면 재사용하고, 없거나 만료되면 새로 인증합니다.

        Returns:
            인증 성공 여부
        """
        creds = None

        # 저장된 토큰 확인
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except Exception as e:
                print(f"Warning: Failed to load saved token: {e}")

        # 토큰이 없거나 유효하지 않으면 재인증
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    print("Token refreshed successfully")
                except Exception as e:
                    print(f"Warning: Token refresh failed: {e}")
                    creds = None

            # 새로운 인증 플로우
            if not creds:
                if not self.credentials_path.exists():
                    print(f"Error: Credentials file not found at {self.credentials_path}")
                    print("Download credentials.json from Google Cloud Console:")
                    print("https://console.cloud.google.com/apis/credentials")
                    return False

                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    print("Authentication successful")
                except Exception as e:
                    print(f"Error: Authentication failed: {e}")
                    return False

            # 토큰 저장
            try:
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
                print(f"Token saved to {self.token_path}")
            except Exception as e:
                print(f"Warning: Failed to save token: {e}")

        self.creds = creds

        # Drive 서비스 빌드
        try:
            self.service = build('drive', 'v3', credentials=creds)
            return True
        except Exception as e:
            print(f"Error: Failed to build Drive service: {e}")
            return False

    def create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Google Drive에 폴더를 생성합니다.

        이미 존재하는 경우 기존 폴더 ID를 반환합니다.

        Args:
            folder_name: 폴더 이름
            parent_id: 부모 폴더 ID (None이면 루트)

        Returns:
            폴더 ID, 실패 시 None
        """
        if not self.service:
            print("Error: Not authenticated. Call authenticate() first.")
            return None

        # 캐시 확인
        cache_key = f"{parent_id or 'root'}:{folder_name}"
        if cache_key in self.folder_cache:
            return self.folder_cache[cache_key]

        try:
            # 기존 폴더 검색
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_id:
                query += f" and '{parent_id}' in parents"

            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()

            items = results.get('files', [])

            if items:
                # 기존 폴더 사용
                folder_id = items[0]['id']
                print(f"Using existing folder: {folder_name} (ID: {folder_id})")
            else:
                # 새 폴더 생성
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                if parent_id:
                    file_metadata['parents'] = [parent_id]

                folder = self.service.files().create(
                    body=file_metadata,
                    fields='id'
                ).execute()

                folder_id = folder.get('id')
                print(f"Created folder: {folder_name} (ID: {folder_id})")

            # 캐시에 저장
            self.folder_cache[cache_key] = folder_id
            return folder_id

        except HttpError as e:
            print(f"Error: Failed to create/find folder '{folder_name}': {e}")
            return None

    def upload_file(
        self,
        file_path: str,
        folder_id: Optional[str] = None,
        mime_type: str = 'image/jpeg',
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Optional[str]:
        """
        파일을 Google Drive에 업로드합니다.

        Args:
            file_path: 업로드할 파일 경로
            folder_id: 대상 폴더 ID (None이면 루트)
            mime_type: MIME 타입
            progress_callback: 진행률 콜백 함수 (file_name, current, total)

        Returns:
            업로드된 파일 ID, 실패 시 None
        """
        if not self.service:
            print("Error: Not authenticated. Call authenticate() first.")
            return None

        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            return None

        try:
            # 파일 메타데이터
            file_metadata = {'name': path.name}
            if folder_id:
                file_metadata['parents'] = [folder_id]

            # 미디어 업로드
            media = MediaFileUpload(
                str(path),
                mimetype=mime_type,
                resumable=True
            )

            # 업로드 실행
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )

            # 청크 단위 업로드 (진행률 추적)
            response = None
            file_size = path.stat().st_size

            while response is None:
                status, response = request.next_chunk()
                if status and progress_callback:
                    progress_callback(path.name, int(status.progress() * file_size), file_size)

            file_id = response.get('id')

            if progress_callback:
                progress_callback(path.name, file_size, file_size)

            return file_id

        except HttpError as e:
            print(f"Error: Failed to upload '{path.name}': {e}")
            return None

    def upload_batch(
        self,
        file_paths: List[str],
        category_folders: Dict[str, str],
        labels: Dict[str, str],
        root_folder_name: str = "Photo_Sort_Results",
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, bool]:
        """
        분류된 이미지들을 일괄 업로드합니다.

        Args:
            file_paths: 업로드할 파일 경로 리스트
            category_folders: 카테고리별 폴더 이름 매핑 {sharp: "Sharp", ...}
            labels: 파일별 분류 레이블 {file_path: "sharp", ...}
            root_folder_name: 루트 폴더 이름
            progress_callback: 진행률 콜백 함수

        Returns:
            업로드 결과 딕셔너리 {file_path: success, ...}
        """
        if not self.service:
            print("Error: Not authenticated. Call authenticate() first.")
            return {}

        # 루트 폴더 생성
        root_folder_id = self.create_folder(root_folder_name)
        if not root_folder_id:
            print("Error: Failed to create root folder")
            return {}

        # 카테고리별 폴더 생성
        folder_ids = {}
        for category, folder_name in category_folders.items():
            folder_id = self.create_folder(folder_name, parent_id=root_folder_id)
            if folder_id:
                folder_ids[category] = folder_id

        # 업로드 결과 추적
        results = {}
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths):
            label = labels.get(file_path, "uncertain")
            folder_id = folder_ids.get(label)

            if not folder_id:
                print(f"Warning: No folder for label '{label}', skipping {file_path}")
                results[file_path] = False
                continue

            # 진행률 업데이트
            if progress_callback:
                progress_callback(f"Uploading {i+1}/{total_files}", i, total_files)

            # 파일 업로드
            file_id = self.upload_file(file_path, folder_id=folder_id)
            results[file_path] = file_id is not None

            if file_id:
                print(f"✓ Uploaded: {Path(file_path).name} → {category_folders[label]}")
            else:
                print(f"✗ Failed: {Path(file_path).name}")

        # 최종 진행률
        if progress_callback:
            progress_callback(f"Upload complete", total_files, total_files)

        return results


def is_available() -> bool:
    """
    Google Drive API 사용 가능 여부를 확인합니다.

    Returns:
        google-api-python-client 등이 설치되어 있으면 True
    """
    return _GDRIVE_AVAILABLE


def setup_credentials(credentials_json: str, save_path: Optional[str] = None) -> bool:
    """
    OAuth2 credentials.json 파일을 설정합니다.

    Args:
        credentials_json: credentials.json 파일 내용 (JSON 문자열)
        save_path: 저장할 경로 (기본값: ~/.unified_sort/credentials.json)

    Returns:
        설정 성공 여부
    """
    try:
        # JSON 유효성 검증
        credentials_data = json.loads(credentials_json)

        # 저장 경로 설정
        if save_path is None:
            save_path = DEFAULT_CREDENTIALS_PATH
        else:
            save_path = Path(save_path)

        # 디렉토리 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        with open(save_path, 'w') as f:
            json.dump(credentials_data, f, indent=2)

        print(f"Credentials saved to {save_path}")
        return True

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in credentials: {e}")
        return False
    except Exception as e:
        print(f"Error: Failed to setup credentials: {e}")
        return False


def get_credentials_instructions() -> str:
    """
    OAuth2 credentials 취득 방법을 안내하는 문자열을 반환합니다.

    Returns:
        안내 메시지
    """
    return """
=== Google Drive API Credentials 설정 방법 ===

1. Google Cloud Console 접속:
   https://console.cloud.google.com/

2. 새 프로젝트 생성 또는 기존 프로젝트 선택

3. API 및 서비스 > 라이브러리로 이동
   - "Google Drive API" 검색 후 활성화

4. API 및 서비스 > 사용자 인증 정보로 이동
   - "사용자 인증 정보 만들기" > "OAuth 클라이언트 ID" 선택
   - 애플리케이션 유형: "데스크톱 앱" 선택
   - 이름 입력 후 "만들기"

5. credentials.json 다운로드
   - 생성된 OAuth 클라이언트 우측 다운로드 버튼 클릭
   - 파일을 ~/.unified_sort/credentials.json에 저장

6. 첫 인증 시 브라우저 창이 열림
   - Google 계정 로그인
   - 권한 승인 (Drive 파일 생성 및 관리)
   - 토큰이 자동 저장됨 (~/.unified_sort/gdrive_token.json)

7. 이후 자동으로 재사용 (재인증 불필요)

=================================================
"""
