# Set PACKAGENAME to the name of your python package

PACKAGENAME		=		projections

install:
	sed -i 's/packagename=\"\"/packagename=\"${PACKAGENAME}\"/' setup.py
	pip install -r requirements.txt
	pip install -e .

pytest:
	python -m pytest test
