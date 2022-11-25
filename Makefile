SHELL := /bin/bash

af: autoformat  ## Alias for `autoformat`
autoformat:  ## Run the autoformatter.
	isort --atomic --profile black  .
	black .
