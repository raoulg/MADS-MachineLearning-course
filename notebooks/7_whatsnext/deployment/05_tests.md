# Tests

When working in production, you dont want a change in your code to break other stuff.
Maybe you will know what could break your code, but you cant expect for all other programmers to know all details of breaking your code...

But you can automate this with tests!

A test is a small piece of code that tests a small part of your code.
Your test should be small and specific. E.g., testing a complete Datahandler class won't help you much: if the test breaks, you don't have a lot more information.

```python
# content of test_sample.py
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

And the test will tell you:

```python
$ pytest
=========================== test session starts ============================
platform linux -- Python 3.x.y, pytest-7.x.y, pluggy-1.x.y
rootdir: /home/sweet/project
collected 1 item

test_sample.py F                                                     [100%]

================================= FAILURES =================================
_______________________________ test_answer ________________________________

    def test_answer():
>       assert inc(3) == 5
E       assert 4 == 5
E        +  where 4 = inc(3)

test_sample.py:6: AssertionError
========================= short test summary info ==========================
FAILED test_sample.py::test_answer - assert 4 == 5
============================ 1 failed in 0.12s =============================
```

Again, you can group your tests in classes:

```python
# content of test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert "e" in x
```

If you add the execution of your tests as part of the `Makefile` you make sure all your tests are done at specific events. 

You can read more about tests in the documentation from [pytest](https://docs.pytest.org/en/7.1.x/contents.html)

