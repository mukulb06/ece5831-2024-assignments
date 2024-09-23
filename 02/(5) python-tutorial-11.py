# main.py
import pricing
from pricing import *
from product import *

net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)



tax = get_tax(100)
print(tax)


import sys

for path in sys.path:
    print(path)

import billing

import sales.pricing
import sales.product

net_price = pricing.get_net_price(
    price=120,
    tax_rate=0.12
)
print(net_price)

tax = get_tax(190)
print(tax)