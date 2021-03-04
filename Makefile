PYTHON_MODULE_PATH=fatima

clean:
	find ${PYTHON_MODULE_PATH} -name "*.pyc" -type f -delete
	find ${PYTHON_MODULE_PATH} -name "__pycache__" -type d -delete
	find ${PYTHON_MODULE_PATH} -name ".ipynb_checkpoints" -type d -delete

format:
	yapf --verbose --in-place --recursive ${PYTHON_MODULE_PATH} --style='{based_on_style: google, indent_width:2, column_limit:80}'
	isort --verbose --force-single-line-imports ${PYTHON_MODULE_PATH}
	docformatter --in-place --recursive ${PYTHON_MODULE_PATH}
