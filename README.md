## Basic Installation and Quickstart
```
cd HEBO
pip install -e .
cd ../GPBT
pip install -r requirements.txt

python scheduler.py  # Train MNIST with HEBO, work in progress.
```

## Protobuf Error
You may need to downgrade the Protobuf package.
```
pip install protobuf~=3.20.0
```