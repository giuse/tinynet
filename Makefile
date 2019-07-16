run: clean tests

delete_pipenv:
	rm Pipfile.lock
	pipenv --rm

init_pipenv:
	pipenv lock
	pipenv sync

reset_pipenv: delete_pipenv init_pipenv

# TODO: 	pipenv run python -m unittest discover -s test
tests:
	pipenv run python -m unittest test

clean:
	find . -name '*.pyc' -delete

.PHONY: init_pipenv clean tests
