# -*- coding: utf-8 -*-
"""
이미지 유사도 기반 분류 시스템 (개선된 사용성 버전)

특징:
- 대화형 메뉴로 쉬운 사용
- 프리뷰 및 확인 기능
- 진행률 표시
- 자동 백업
- 상세한 리포트

간단 사용법:
    python image_sorter.py
    (그 다음 메뉴에서 선택)

고급 사용법:
    python image_sorter.py --source ./photos --output ./sorted --threshold 10
"""

import os
import argparse
import shutil
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional


class ProgressBar:
    """작업 진행률을 시각적으로 보여주는 클래스입니다."""
    
    def __init__(self, total: int, prefix: str = '', length: int = 50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: int = None):
        """진행률을 업데이트하고 화면에 표시합니다."""
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(self.length * percent)
        bar = '█' * filled + '░' * (self.length - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{int(eta)}초 남음"
        else:
            eta_str = "계산 중..."
        
        print(f'\r{self.prefix} |{bar}| {percent*100:.1f}% ({self.current}/{self.total}) - {eta_str}', 
              end='', flush=True)
        
        if self.current >= self.total:
            print()


class ImageHasher:
    """이미지의 perceptual hash를 계산하는 클래스입니다."""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
    
    def calculate_hash(self, image_path: str) -> Optional[int]:
        """이미지 파일의 perceptual hash를 계산합니다."""
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            diff = pixels[:, 1:] < pixels[:, :-1]
            hash_value = 0
            for i, bit in enumerate(diff.flatten()):
                if bit:
                    hash_value |= (1 << i)
            return hash_value
        except Exception as e:
            print(f"\n⚠️  경고: {Path(image_path).name} 해시 계산 실패 - {e}")
            return None
    
    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """두 해시값 사이의 Hamming distance를 계산합니다."""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance


class ImageGrouper:
    """이미지들을 유사도 기준으로 그룹화하는 클래스입니다."""
    
    def __init__(self, threshold: int = 8):
        self.hasher = ImageHasher()
        self.threshold = threshold
    
    def scan_images(self, source_dir: str, extensions: List[str] = None) -> List[Dict]:
        """지정된 폴더에서 이미지를 찾고 해시를 계산합니다."""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png']
        
        images = []
        source_path = Path(source_dir)
        
        print(f"\n🔍 {source_dir} 폴더 스캔 중...\n")
        
        all_files = []
        for ext in extensions:
            all_files.extend(list(source_path.rglob(f'*{ext}')))
            all_files.extend(list(source_path.rglob(f'*{ext.upper()}')))
        
        all_files = [f for f in all_files if f.is_file()]
        
        if len(all_files) == 0:
            print("❌ 이미지를 찾을 수 없습니다.")
            return images
        
        print(f"총 {len(all_files)}개 이미지 파일 발견")
        print("해시 계산 중...\n")
        
        progress = ProgressBar(len(all_files), prefix='진행률')
        
        for file_path in all_files:
            hash_value = self.hasher.calculate_hash(str(file_path))
            if hash_value is not None:
                images.append({
                    'path': file_path,
                    'hash': hash_value,
                    'name': file_path.name,
                    'stem': file_path.stem,
                    'size': file_path.stat().st_size
                })
            progress.update()
        
        print(f"\n✅ {len(images)}개 이미지 처리 완료\n")
        return images
    
    def preview_groups(self, groups: List[List[Dict]], max_preview: int = 5) -> None:
        """그룹화 결과를 미리 보여줍니다."""
        print("\n" + "="*70)
        print("📊 그룹화 결과 미리보기")
        print("="*70 + "\n")
        
        total_images = sum(len(group) for group in groups)
        print(f"발견된 그룹: {len(groups)}개")
        print(f"그룹에 속한 이미지: {total_images}개\n")
        
        for i, group in enumerate(groups[:max_preview], start=1):
            print(f"📁 그룹 {i} ({len(group)}개 이미지)")
            print("─" * 70)
            
            for img in group[:3]:
                size_mb = img['size'] / (1024 * 1024)
                print(f"  • {img['name']} ({size_mb:.2f} MB)")
            
            if len(group) > 3:
                print(f"  ... 외 {len(group) - 3}개 파일")
            print()
        
        if len(groups) > max_preview:
            remaining = len(groups) - max_preview
            remaining_images = sum(len(group) for group in groups[max_preview:])
            print(f"... 외 {remaining}개 그룹 ({remaining_images}개 이미지)\n")
    
    def group_similar_images(self, images: List[Dict], show_progress: bool = True) -> List[List[Dict]]:
        """이미지들을 유사도 기준으로 그룹화합니다."""
        if show_progress:
            print(f"\n🔗 유사한 이미지 그룹화 중... (임계값: {self.threshold})\n")
        
        used = set()
        groups = []
        
        if show_progress:
            progress = ProgressBar(len(images), prefix='그룹화')
        
        for i, img in enumerate(images):
            if i in used:
                if show_progress:
                    progress.update()
                continue
            
            group = [img]
            used.add(i)
            
            for j, other_img in enumerate(images[i+1:], start=i+1):
                if j in used:
                    continue
                
                distance = self.hasher.hamming_distance(img['hash'], other_img['hash'])
                
                if distance <= self.threshold:
                    group.append(other_img)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
            
            if show_progress:
                progress.update()
        
        groups.sort(key=len, reverse=True)
        
        if show_progress:
            print(f"\n✅ {len(groups)}개 그룹 발견\n")
        
        return groups
    
    def save_groups(self, groups: List[List[Dict]], output_dir: str, move: bool = False) -> Dict:
        """그룹화된 이미지들을 폴더별로 저장합니다."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_groups': len(groups),
            'total_files': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'groups_info': []
        }
        
        print(f"\n💾 그룹별 폴더에 저장 중...")
        print(f"저장 위치: {output_dir}")
        print(f"모드: {'이동' if move else '복사'}\n")
        
        total_files = sum(len(group) for group in groups)
        progress = ProgressBar(total_files, prefix='저장 중')
        
        for group_idx, group in enumerate(groups, start=1):
            group_folder = output_path / f"group_{group_idx:03d}"
            group_folder.mkdir(exist_ok=True)
            
            group_info = {'name': f"group_{group_idx:03d}", 'count': len(group), 'files': []}
            
            for img in group:
                dest_path = group_folder / img['name']
                try:
                    if dest_path.exists():
                        stats['skipped'] += 1
                    else:
                        if move:
                            shutil.move(str(img['path']), str(dest_path))
                        else:
                            shutil.copy2(str(img['path']), str(dest_path))
                        stats['success'] += 1
                        group_info['files'].append(img['name'])
                except Exception as e:
                    stats['failed'] += 1
                    print(f"\n⚠️  오류: {img['name']} - {e}")
                progress.update()
            
            stats['groups_info'].append(group_info)
        
        stats['total_files'] = total_files
        self._save_report(stats, output_path)
        return stats
    
    def _save_report(self, stats: Dict, output_path: Path) -> None:
        """작업 결과 리포트를 저장합니다."""
        report_file = output_path / 'classification_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("이미지 분류 작업 리포트\n")
            f.write("="*70 + "\n\n")
            f.write(f"작업 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 그룹 수: {stats['total_groups']}\n")
            f.write(f"처리된 파일: {stats['total_files']}개\n")
            f.write(f"  - 성공: {stats['success']}개\n")
            f.write(f"  - 실패: {stats['failed']}개\n")
            f.write(f"  - 건너뜀: {stats['skipped']}개\n\n")
            f.write("="*70 + "\n그룹별 상세 정보\n" + "="*70 + "\n\n")
            for group_info in stats['groups_info']:
                f.write(f"📁 {group_info['name']} ({group_info['count']}개 파일)\n")
                for file_name in group_info['files']:
                    f.write(f"  • {file_name}\n")
                f.write("\n")


class RawFileSyncer:
    """JPG 파일의 위치에 맞춰 RW2(RAW) 파일을 동기화하는 클래스입니다."""
    
    def __init__(self, raw_extensions: List[str] = None):
        if raw_extensions is None:
            self.raw_extensions = ['.rw2', '.raw', '.cr2', '.nef', '.arw', '.dng', '.orf']
        else:
            self.raw_extensions = raw_extensions
    
    def find_jpg_files(self, sorted_dir: str) -> Dict[str, Path]:
        """정리된 폴더에서 모든 JPG 파일을 찾습니다."""
        jpg_map = {}
        sorted_path = Path(sorted_dir)
        print(f"\n🔍 정리된 JPG 파일 검색 중...\n")
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            for jpg_file in sorted_path.rglob(f'*{ext}'):
                if jpg_file.is_file():
                    jpg_map[jpg_file.stem] = jpg_file
        print(f"✅ {len(jpg_map)}개 JPG 파일 발견\n")
        return jpg_map
    
    def find_raw_files(self, source_dir: str) -> Dict[str, Path]:
        """원본 폴더에서 모든 RAW 파일을 찾습니다."""
        raw_map = {}
        source_path = Path(source_dir)
        print(f"🔍 원본 폴더에서 RAW 파일 검색 중...\n")
        for ext in self.raw_extensions:
            for raw_file in source_path.rglob(f'*{ext}'):
                if raw_file.is_file():
                    raw_map[raw_file.stem] = raw_file
            for raw_file in source_path.rglob(f'*{ext.upper()}'):
                if raw_file.is_file():
                    raw_map[raw_file.stem] = raw_file
        print(f"✅ {len(raw_map)}개 RAW 파일 발견\n")
        return raw_map
    
    def sync_raw_files(self, source_dir: str, sorted_dir: str, move: bool = False) -> Dict:
        """JPG의 위치에 맞춰 RAW 파일을 동기화합니다."""
        jpg_map = self.find_jpg_files(sorted_dir)
        raw_map = self.find_raw_files(source_dir)
        
        matched = [(stem, jpg_path, raw_map[stem]) for stem, jpg_path in jpg_map.items() if stem in raw_map]
        stats = {'total': len(matched), 'success': 0, 'failed': 0, 'skipped': 0}
        
        if len(matched) == 0:
            print("⚠️  동기화할 RAW 파일이 없습니다.\n")
            return stats
        
        print(f"💾 RAW 파일 동기화 시작... (모드: {'이동' if move else '복사'})\n")
        progress = ProgressBar(len(matched), prefix='동기화')
        
        for stem, jpg_path, raw_path in matched:
            dest_path = jpg_path.parent / raw_path.name
            try:
                if dest_path.exists():
                    stats['skipped'] += 1
                else:
                    if move:
                        shutil.move(str(raw_path), str(dest_path))
                    else:
                        shutil.copy2(str(raw_path), str(dest_path))
                    stats['success'] += 1
            except Exception as e:
                stats['failed'] += 1
                print(f"\n⚠️  오류: {raw_path.name} - {e}")
            progress.update()
        
        return stats


class InteractiveMenu:
    """대화형 메뉴 시스템을 제공하는 클래스입니다."""
    
    def __init__(self):
        self.grouper = None
        self.syncer = RawFileSyncer()
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        print("\n" + "="*70)
        print(" "*20 + "📷 이미지 유사도 분류 도구")
        print("="*70 + "\n")
    
    def show_main_menu(self) -> str:
        self.show_header()
        print("작업을 선택하세요:\n")
        print("  1. JPG 파일 분류 (유사한 이미지 그룹화)")
        print("  2. RAW 파일 동기화 (분류된 JPG에 맞춰)")
        print("  3. 전체 작업 (분류 + 동기화)")
        print("  4. 종료\n")
        return input("선택 (1-4): ").strip()
    
    def get_folder_path(self, prompt: str, must_exist: bool = True) -> Optional[str]:
        while True:
            path = input(f"\n{prompt}: ").strip().strip('"\'')
            if not path:
                print("❌ 경로를 입력해주세요.")
                continue
            path_obj = Path(path)
            if must_exist and not path_obj.exists():
                print(f"❌ 폴더를 찾을 수 없습니다: {path}")
                if input("다시 입력하시겠습니까? (y/n): ").strip().lower() != 'y':
                    return None
                continue
            return str(path_obj.absolute())
    
    def get_threshold(self) -> int:
        print("\n유사도 임계값을 설정하세요:")
        print("  - 낮은 값 (5-7): 거의 동일한 이미지만")
        print("  - 중간 값 (8-12): 권장 설정")
        print("  - 높은 값 (13-20): 약간 다른 이미지도 포함")
        while True:
            try:
                threshold = int(input("\n임계값 (기본값: 10): ").strip() or "10")
                if 0 <= threshold <= 30:
                    return threshold
                print("❌ 0에서 30 사이의 값을 입력하세요.")
            except ValueError:
                print("❌ 숫자를 입력하세요.")
    
    def confirm_action(self, message: str) -> bool:
        return input(f"\n{message} (y/n): ").strip().lower() == 'y'
    
    def run_classification(self):
        self.clear_screen()
        self.show_header()
        print("📸 JPG 파일 분류 작업\n" + "="*70 + "\n")
        
        source = self.get_folder_path("원본 이미지 폴더 경로", must_exist=True)
        if not source:
            return
        output = self.get_folder_path("분류 결과 저장 폴더 경로", must_exist=False)
        if not output:
            return
        threshold = self.get_threshold()
        
        print("\n파일 처리 방식:")
        print("  1. 복사 (원본 유지, 안전)")
        print("  2. 이동 (원본 이동, 빠름)")
        move = (input("\n선택 (1-2, 기본값: 1): ").strip() or "1") == "2"
        
        print("\n" + "="*70)
        print(f"설정 확인:\n  원본 폴더: {source}\n  출력 폴더: {output}\n  임계값: {threshold}\n  모드: {'이동' if move else '복사'}")
        print("="*70)
        
        if not self.confirm_action("이 설정으로 진행하시겠습니까?"):
            print("\n❌ 작업이 취소되었습니다.")
            input("\n계속하려면 Enter를 누르세요...")
            return
        
        try:
            self.grouper = ImageGrouper(threshold=threshold)
            images = self.grouper.scan_images(source)
            if len(images) == 0:
                print("\n❌ 처리할 이미지가 없습니다.")
                input("\n계속하려면 Enter를 누르세요...")
                return
            
            groups = self.grouper.group_similar_images(images)
            if len(groups) == 0:
                print("\n⚠️  유사한 이미지를 찾지 못했습니다.")
                input("\n계속하려면 Enter를 누르세요...")
                return
            
            self.grouper.preview_groups(groups)
            if not self.confirm_action("이대로 저장하시겠습니까?"):
                print("\n❌ 작업이 취소되었습니다.")
                input("\n계속하려면 Enter를 누르세요...")
                return
            
            stats = self.grouper.save_groups(groups, output, move=move)
            print("\n" + "="*70 + "\n✅ 작업 완료!\n" + "="*70)
            print(f"\n총 {stats['total_groups']}개 그룹이 생성되었습니다.")
            print(f"처리된 파일: {stats['success']}개")
            if stats['failed'] > 0:
                print(f"실패: {stats['failed']}개")
            if stats['skipped'] > 0:
                print(f"건너뜀: {stats['skipped']}개")
            print(f"\n결과 위치: {output}\n리포트 파일: {output}/classification_report.txt\n")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        input("\n계속하려면 Enter를 누르세요...")
    
    def run_sync(self):
        self.clear_screen()
        self.show_header()
        print("📷 RAW 파일 동기화 작업\n" + "="*70 + "\n")
        
        source = self.get_folder_path("원본 RAW 파일 폴더 경로", must_exist=True)
        if not source:
            return
        sorted_dir = self.get_folder_path("정리된 JPG 파일 폴더 경로", must_exist=True)
        if not sorted_dir:
            return
        
        print("\n파일 처리 방식:\n  1. 복사 (원본 유지, 안전)\n  2. 이동 (원본 이동, 빠름)")
        move = (input("\n선택 (1-2, 기본값: 1): ").strip() or "1") == "2"
        
        try:
            stats = self.syncer.sync_raw_files(source, sorted_dir, move=move)
            print("\n" + "="*70 + "\n✅ 동기화 완료!\n" + "="*70)
            print(f"\n처리된 파일: {stats['success']}/{stats['total']}개")
            if stats['failed'] > 0:
                print(f"실패: {stats['failed']}개")
            if stats['skipped'] > 0:
                print(f"건너뜀: {stats['skipped']}개\n")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        input("\n계속하려면 Enter를 누르세요...")
    
    def run_full_workflow(self):
        self.clear_screen()
        self.show_header()
        print("🚀 전체 작업 (분류 + 동기화)\n" + "="*70 + "\n")
        
        source = self.get_folder_path("원본 폴더 경로 (JPG와 RAW 모두 포함)", must_exist=True)
        if not source:
            return
        output = self.get_folder_path("분류 결과 저장 폴더 경로", must_exist=False)
        if not output:
            return
        threshold = self.get_threshold()
        
        print("\n파일 처리 방식:\n  1. 복사 (원본 유지, 안전)\n  2. 이동 (원본 이동, 빠름)")
        move = (input("\n선택 (1-2, 기본값: 1): ").strip() or "1") == "2"
        
        print("\n" + "="*70)
        print(f"설정 확인:\n  원본 폴더: {source}\n  출력 폴더: {output}\n  임계값: {threshold}\n  모드: {'이동' if move else '복사'}")
        print("="*70)
        
        if not self.confirm_action("이 설정으로 진행하시겠습니까?"):
            print("\n❌ 작업이 취소되었습니다.")
            input("\n계속하려면 Enter를 누르세요...")
            return
        
        try:
            print("\n" + "="*70 + "\n1단계: JPG 파일 분류\n" + "="*70)
            self.grouper = ImageGrouper(threshold=threshold)
            images = self.grouper.scan_images(source)
            if len(images) == 0:
                print("\n❌ 처리할 이미지가 없습니다.")
                input("\n계속하려면 Enter를 누르세요...")
                return
            
            groups = self.grouper.group_similar_images(images)
            if len(groups) == 0:
                print("\n⚠️  유사한 이미지를 찾지 못했습니다.")
                input("\n계속하려면 Enter를 누르세요...")
                return
            
            stats1 = self.grouper.save_groups(groups, output, move=move)
            
            print("\n" + "="*70 + "\n2단계: RAW 파일 동기화\n" + "="*70)
            stats2 = self.syncer.sync_raw_files(source, output, move=move)
            
            print("\n" + "="*70 + "\n✅ 전체 작업 완료!\n" + "="*70)
            print(f"\nJPG 분류: {stats1['success']}개 파일, {stats1['total_groups']}개 그룹")
            print(f"RAW 동기화: {stats2['success']}/{stats2['total']}개 파일")
            print(f"\n결과 위치: {output}\n리포트 파일: {output}/classification_report.txt\n")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        input("\n계속하려면 Enter를 누르세요...")
    
    def run(self):
        """메인 루프를 실행합니다."""
        while True:
            choice = self.show_main_menu()
            if choice == '1':
                self.run_classification()
            elif choice == '2':
                self.run_sync()
            elif choice == '3':
                self.run_full_workflow()
            elif choice == '4':4
            else:
                print("\n❌ 잘못된 선택입니다.")
                input("\n계속하려면 Enter를 누르세요...")


def main():
    parser = argparse.ArgumentParser(description='이미지 유사도 기반 분류 시스템')
    parser.add_argument('--source', help='원본 이미지 폴더')
    parser.add_argument('--output', help='분류 결과 저장 폴더')
    parser.add_argument('--threshold', type=int, default=10, help='유사도 임계값 (기본값: 10)')
    parser.add_argument('--move', action='store_true', help='복사 대신 이동')
    parser.add_argument('--sync-only', action='store_true', help='RAW 파일 동기화만 실행')
    args = parser.parse_args()
    
    if args.source and args.output:
        # 커맨드라인 모드
        if args.sync_only:
            print("\n📷 RAW 파일 동기화 모드\n")
            syncer = RawFileSyncer()
            stats = syncer.sync_raw_files(args.source, args.output, move=args.move)
            print(f"\n처리 완료: {stats['success']}/{stats['total']}개")
        else:
            print("\n📸 이미지 분류 모드\n")
            grouper = ImageGrouper(threshold=args.threshold)
            images = grouper.scan_images(args.source)
            if images:
                groups = grouper.group_similar_images(images)
                if groups:
                    stats = grouper.save_groups(groups, args.output, move=args.move)
                    print(f"\n처리 완료: {stats['success']}개 파일, {stats['total_groups']}개 그룹")
    else:
        # 대화형 모드
        menu = InteractiveMenu()
        menu.run()


if __name__ == '__main__':
    main()