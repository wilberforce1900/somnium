"""pytest 根配置：仓库根入 sys.path，使 `import src` 在任意 cwd 下可用。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
