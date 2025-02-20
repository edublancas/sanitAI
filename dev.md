
```sh
docker exec -it presidioui /venvs/appenv/bin/python -m presidioui.models load

docker exec -it presidioui /venvs/appenv/bin/python -m presidioui.models delete
```

## Development

~~~bash
conda create --name presidioui python=3.12 -c conda-forge -y
conda activate presidioui
pip install -r requirements.txt
pip install -e presidioui/
~~~


~~~bash
python -m spacy download en_core_web_lg

python -m presidioui.ui ui
python -m presidioui.proxy

# load json models
python -m presidioui.models load
# delete all rules and samples
python -m presidioui.models delete
~~~

The application will be available at `http://localhost:5000`



```sh
# RUN_FIXER example: some test acses cases are failing
python -m presidioui.ai fixer rules/fixer-example-visa-detector.json

# RUN_FIXER EXAMPLE: this is impossible to solve since it contains the same test,
# in one case it says it should trigger the rule, in the second one it says it should not
python -m presidioui.ai fixer rules/fixer-example-visa-detector-with-incorrect-test.json --max-attempts 1


# improve workflow
# to test EDIT_TEST_CASES we can do: "add more test cases"
python -m presidioui.ai improve rules/visa-detector-all-passing.json

# to test MODIFY_RULE we can do "add support for master card cards"
python -m presidioui.ai improve rules/visa-detector-all-passing.json

# to test RUN_FIXER
python -m presidioui.ai improve rules/fixer-example-visa-detector.json

```
