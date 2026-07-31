"""Apply the fixed MuScriptor best-quality runtime patch after installation."""

from src.utils.muscriptor_source_patch import apply_muscriptor_quality_patch


def main() -> None:
    apply_muscriptor_quality_patch()


if __name__ == "__main__":
    main()
