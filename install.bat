set env_name=ampligraph

call conda create -y -n %env_name% python=3.6
call conda activate %env_name%

REM To generate conda-spec-file.txt just run 'conda list --explicit > conda-spec-file.txt' in the target directory
call conda install --name %env_name% --file conda-spec-file.txt

REM ###FOMM###
REM To generate pip-spec-file.txt just run 'pip freeze > pip-spec-file.txt' in the target directory
REM call pip install -r pip-spec-file.txt