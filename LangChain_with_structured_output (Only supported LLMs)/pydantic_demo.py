# Pydantic is a data validation and parsing library for Python. It ensures that the data we work with is correct, structured and type-safe
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Student(BaseModel):
    name: str
    email: Optional[EmailStr] = None # Default validation of pydantic (for emails)
    age: int
    score: float = 0.00 # default value
    cgpa: Optional[float] = None # this None is important for setting the default value
    percentage: Optional[float] = Field(default=None, gt=0, lt=100, description='Percentage of the student in higher secondary examination...') # Constraint: default value None (since optional, if not optional, it can be other values as well), else greater than 0 and less than 100, other options are :-
    # gt: > , ge: >= , lt: < , le: <= , multiple_of: % (divisible by), max_length/min_length -> string length limits, pattern -> regex match, max_digits/decimal_places -> decimal constraints

s1 = {'name': 'souryadip', 'age': 20, 'score': 92.5}
s1 = Student(**s1)
print(s1) # the returned type of object is a pydantic object, so we need to convert it to regular python dictionary or json (shown in the later examples)

s2 = {'name': 'souryadip', 'age': '20', 'score': 92} # Automatic type coercion occurs here. If the coercion or type conversion is possible, pydantic automatically does that. For eg, here '20' is string but is convertible to int format. So pydantic does it for us automatically.
s2 = Student(**s2)
print(s2)

# s3 = {'name': 'souryadip', 'age': 'twenty', 'score': 92}
# s3 = Student(**s3)
# print(s3) ----> Error [Conversion to int not possible]

s4 = {'name': 'souryadip', 'age': 20, 'cgpa': 9.83}
s4 = Student(**s4)
print(s4)

# s5 = {'name': 'souryadip', 'age': 20, 'cgpa': 9.83, 'email': 'abc'}
# s5 = Student(**s5)
# print(s5) ---> Error [Invalid email]

s5 = {'name': 'souryadip', 'age': 20, 'cgpa': 9.83, 'email': 'souryadip@gmail.com'}
s5 = Student(**s5)
print(s5)

s6 = {'name': 'souryadip', 'age': 20, 'cgpa': 9.83, 'email': 'souryadip@gmail.com', 'percentage': 98.3}
s6 = Student(**s6)
print(dict(s6))
print(s6.model_dump_json()) # To convert to JSON