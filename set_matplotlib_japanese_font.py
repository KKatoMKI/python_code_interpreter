from pathlib import Path
import matplotlib as mpl

# 設定ディレクトリの作成
config_dir = Path(mpl.get_configdir())
config_dir.mkdir(exist_ok=True, parents=True)

# デフォルトの設定ファイルパスとコピー先パス
default_config_path = Path(mpl.__file__).parent / "mpl-data/matplotlibrc"
config_path = config_dir / "matplotlibrc"
config_path = config_dir / "matplotlibrc"
print(f"Default config path: {default_config_path}")
print(f"Config path: {config_path}")

# 内容の変更と追記処理
font_family_set = False
output_lines = []

with default_config_path.open("r", encoding="utf-8") as src:
    for line in src:
        stripped = line.strip()
        if stripped.startswith("font.family"):
            output_lines.append(f"#{stripped}\n")
        else:
            output_lines.append(line)

output_lines.append("\nfont.family: Migmix 1p\n")
# output_lines.append("\nfont.family: IPAGothic\n")

# ファイルに書き込み
with config_path.open("w", encoding="utf-8") as dst:
    dst.writelines(output_lines)

# matplotlibキャッシュ削除
cache_dir = Path(mpl.get_cachedir())
print(f"Cache directory: {cache_dir}")

def remove_tree(path: Path):
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            remove_tree(child)
        else:
            try:
                child.unlink()
            except Exception:
                pass
    try:
        path.rmdir()
    except Exception:
        pass

remove_tree(cache_dir)