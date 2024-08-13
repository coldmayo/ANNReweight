# Using NERSC

This is a simple workflow for running this project using NERSC

## Login:

If you already have an account (and MFA) setup, then you are ready for this next step.

SSH into perlmutter:
```bash
$ ssh your-username@perlmutter.nersc.gov
```
Give password and OTP when prompted:
```bash
your-password123456   # form it should be in
```

Transfer needed files into it (do this from a different terminal window):
```bash
$ scp -r ANNReweight/data your-username@perlmutter.nersc.gov:/global/homes/c/your-username
$ scp -r ANNReweight/model your-username@perlmutter.nersc.gov:/global/homes/c/your-username
```

Set up Python Envirement (in nersc terminal):
```bash
$ module load python/3.9   # Load python 3.9
$ python -m venv venv   # make python virtual envirenment
$ source venv/bin/activate   # activate venv
$ pip install matplotlib numpy tqdm pandas cupy-cuda117   # install some needed packages
```

Run python scripts, for example:
```bash
python -u train_PFN.py 50
```