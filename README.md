# Common Utils

- [Packaging Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

```bash
python3 -m pip install --upgrade build

python3 -m build

python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```

Username: __token__
Password: Token API

```bash
python -m pip install -i https://test.pypi.org/simple/ gaohn-common-utils==0.0.15
```
