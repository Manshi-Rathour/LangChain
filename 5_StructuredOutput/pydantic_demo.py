from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=9)

new_student = {'name': 'manshi', 'age': 23, 'email': 'abc@gmail.com', 'cgpa': 8}
student = Student(**new_student)

print(student)

student_dict = dict(student)
print(student_dict['age'])