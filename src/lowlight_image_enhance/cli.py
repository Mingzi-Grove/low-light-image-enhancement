import argparse
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from . import utils, pipeline


DEFAULT_OUTPUT_DIR = "examples/output"


def main() -> None:
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1. 创建解析器
    parser = argparse.ArgumentParser(description="Low-light image enhancement")

    # 2. 登记 argparse 参数
    # 2.1. input_path: 输入图像路径
    parser.add_argument("input_path", help="Input image path")
    # 2.2. output_dir: 输出图像路径（文件夹），默认值为 examples/output
    parser.add_argument("--output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    # 3. 解析命令行输入的参数
    args = parser.parse_args()

    try:
        # 4. 检查 output_dir 并处理
        in_path = Path(args.input_path)
        out_dir = Path(args.output_dir)
        if not out_dir.is_dir():
            # 目标文件夹不存在就报错
            raise NotADirectoryError(f"Output directory not found: {out_dir}")
        # 路径没有问题就构建完整的输出路径（包含文件名，f"{原名}_enhanced.png"）
        out_path = out_dir / f"{in_path.stem}_enhanced.png"

        # 5. read -> enhance -> write
        img = utils.imread(args.input_path)
        img_enhanced = pipeline.enhance(img)
        utils.imwrite(str(out_path), img_enhanced)
    except Exception:
        logger.exception("An error occurred during processing.")
        raise SystemExit(1)  # 退出程序，返回非零状态码表示错误

# 被当作主程序时执行
if __name__ == "__main__":
    main()