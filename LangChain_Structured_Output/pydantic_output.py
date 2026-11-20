'''
# from langchain_groq import ChatGroq
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name:str
    # sending something as optional
    age:Optional[int] = None #yahan none send krna is imp
    # builtin validation
    email:EmailStr
    # gt = greater than, lt=less than
    cgpa:float = Field(gt = 0, lt = 10, default = 5)


s1 = {"name":"Roy", "email":"ronit@gmail.com", "cgpa":9.01}

student = Student(**s1)
print(student)
student_json = student.model_dump_json()

'''


from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

class Review(BaseModel):
    key_themes: list[str] = Field(description="This is the list of string having the key themes taht is discussed in the product review")
    summary: str = Field(description="this is the brief summary of the product")
    sentiment: Literal["pos", "neg"] = Field(description="Return the sentiment of the review either, neutral, positive, or negative")
    pros: Optional[list[str]] = Field(description="This is the list of pros that is discussed in the product review")
    cons: Optional[list[str]] = Field(description="This is the list of cons that is discussed in the product review")

structured_output = model.with_structured_output(Review)

result = structured_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.""")

print(result)
print("<----------------->")
print("This is the list of pros:", result.pros)