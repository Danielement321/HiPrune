conda create -n HiPrune python=3.12 -y
conda activate HiPrune
cd lmms-eval
pip install av==14.4.0
pip install --no-deps -U -e .
cd ../LLaVA
pip install --no-deps -U -e .
pip install openpyxl==3.1.5 qwen_vl_utils
cd ../lmms-eval/miscs
pip install -r llava_repr_requirements.txt
