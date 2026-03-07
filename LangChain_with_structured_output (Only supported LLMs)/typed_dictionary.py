from typing import TypedDict
class Person(TypedDict):
    name: str # key: data-type format
    age: int

p1: Person = {'name': 'Souryadip', 'age':20}
print(p1)