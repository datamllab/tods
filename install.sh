pip install -r requirements.txt

cd d3m
pip install -e .
cd ..

cd tods/common-primitives
pip install -e .
cd ../..

cd tods/common-primitives/sklearn-wrap
pip install -e .
cd ../../..

cd tods 
pip3 install -e .
cd ..

cd axolotl
pip3 install -e .
pip3 install -e .[cpu]
cd ..

