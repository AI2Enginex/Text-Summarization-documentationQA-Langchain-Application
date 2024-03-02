from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Course
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,  login, logout
from django.contrib.auth import get_user_model
from text_summarization import LongTextSummarisation
from text_summarization import DocumentSummarisation
from text_summarization import DocumentQA
from .forms import UploadFileForm
import os
import cleantext
import json
# Create your views here.

filepath = ''
user_input = ''
class TextAnalysis:

    @classmethod
    def long_text(cls, user_input):

        try:
            t = LongTextSummarisation(model='gpt-3.5-turbo-instruct')
            return t.summarise_long_text(user_input=user_input)
        except Exception as e:
            return e

    @classmethod
    def summarize_doc(cls, filename,user_input):
        try:
            docs = DocumentSummarisation(model='gpt-3.5-turbo-instruct', filepath=filename)
            return docs.summarise_doc_text(delimeter=['.',','],size=1000,overlap=300,user_query=user_input)
        except Exception as e:
            return e
    
    @classmethod
    def question_answer(cls, filename,user_question):

        try:
            que_ = DocumentQA(model='gpt-3.5-turbo-instruct',filepath=filename)
            return que_.run_engine(size=1000,overlap=300,query=user_question)
        except Exception as e:
            return e
    


User = get_user_model()


def index(request):

    return render(request, "home.html")


def summarization(request):

    return render(request, "text_summarization.html")


def doc_summary(request):

    return render(request, "document_summarization.html")


def question_answer(request):

    return render(request, "document_qa.html")


def user_page(request):
    return render(request, 'user_page.html')


def display_courses(request):
    # Filter courses based on the logged-in user
    user_courses = Course.objects.filter(username=request.user.username)
    return render(request, 'user_page.html', {'courses': user_courses})


def delete_account(request):
    return render(request, 'delete_account.html')


def update_password(request):
    return render(request, 'forgot_password.html')


def handleSignUp(request):
    if request.method == "POST":
        # Get the post parameters
        username = request.POST['username']
        email = request.POST['email']
        fname = request.POST['fname']
        lname = request.POST['lname']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        # check for erroneous input
        if len(username) < 10:
            messages.error(
                request, " Your username must be under 10 characters")
            return redirect('/')

        if not username.isalnum():
            messages.error(
                request, " Username should only contain letters and numbers")
            return redirect('/')

        if (pass1 != pass2):
            messages.error(request, " Passwords do not match")
            return redirect('/')

        # Check if the username already exists
        if User.objects.filter(username=username).exists():
            messages.error(
                request, "Username already exists. Choose a different one.")
            return redirect('/')

        # Create the user
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.save()
        messages.success(request, "Your iCoder has been successfully created")
        return redirect('/')

    else:
        return HttpResponse("404 - Not found")


def handleLogin(request):
    if request.method == "POST":
   
        loginusername = request.POST['name']
        loginpassword = request.POST['loginpassword']

   
        try:
            user = User.objects.get(username=loginusername)
        except User.DoesNotExist:
            messages.error(
                request, "User does not exist. Please check your credentials.")
            return redirect("/")

        user = authenticate(username=loginusername, password=loginpassword)
        print(type(user))
        if user is not None:
            login(request, user)
            messages.success(request, "Successfully Logged In")
            return redirect("/")
        else:
            messages.error(request, "Invalid credentials! Please try again")
            return redirect("/")

    return HttpResponse("404- Not found")


def handelLogout(request):
    if request.user.is_authenticated:
        logout(request)
        messages.success(request, "Successfully logged out")
    else:
        messages.warning(request, "You are not logged in")
    return redirect('/')


# def enrool(request):
#     if request.method == 'POST' and request.user.is_authenticated:
#         course_name = request.POST['freecourse'].lower()
#         course_dict = CoursesList.course_details()

#         # Check if the user is already enrolled in the course
#         if Course.objects.filter(username=request.user.username, course_name=course_name).exists():
#             messages.warning(
#                 request, f"You are already enrolled in {course_name}.")
#         elif course_name in course_dict:
#             Course.objects.create(
#                 username=request.user.username,
#                 course_name=course_name,
#                 course_description=course_dict[course_name],
#             )
#             messages.success(request, f"You have enrolled in {course_name} successfully!")
#         else:
#             messages.error(request, "Invalid course selected.")
#     else:
#         messages.warning(request, "You are not logged in")

#     return redirect('courses')


def unenroll_course(request, course_name):
  
    course = get_object_or_404(
        Course, username=request.user.username, course_name=course_name)

    course.delete()
    messages.success(
        request, f"You have unenrolled from {course_name} successfully!")

    return redirect('profile')


def delete_existing_user(request):

    if request.method == 'POST':
        username = request.POST['name']
        password = request.POST['loginpassword']

        existing_user = User.objects.filter(username=username)

        if existing_user.exists():
            user = authenticate(username=username, password=password)
            if user is not None:
                existing_user.delete()
                messages.warning(
                    request, f"User with username '{username}' has been deleted Successfully.")
            else:

                messages.error(
                    request, "Password is incorrect. User not deleted.")
                return redirect("user_delete")

    return redirect('index_view')


def update_user_password(request):
    if request.method == 'POST':
        username = request.POST['name']
        new_password = request.POST['loginpassword']
        validate_password = request.POST['loginpassword1']

        if new_password != validate_password:
            messages.error(request, "password did not match")
            return redirect("user_update")

        try:
            user = User.objects.get(username=username)
            user.set_password(new_password)
            user.save()
            messages.success(request, "Password updated successfully.")
        except User.DoesNotExist:
            messages.error(
                request, f"User with username '{username}' does not exist.")

    return redirect('index_view')


def summarize_long_text(request):

    if request.method == 'POST' and request.user.is_authenticated:

        userinput = request.POST['userinput']
        if userinput == "":
            messages.error(request, "Empty string")
            return redirect("textsummarization")

        summary = TextAnalysis.long_text(user_input=userinput)
        summary = summary.split('\n')
        return render(request, 'text_summarization.html', {'text_summary': summary})
    else:
        messages.error(request, "Please Login first")
        return redirect("textsummarization")


def doc_summary_generator(request):

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            upload_dir = 'E:/react-django/django_sessions/uploads'
            os.makedirs(upload_dir, exist_ok=True)
            fs = FileSystemStorage(location=upload_dir)
            saved_file = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(upload_dir + "/"+saved_file)
            file_name = str(uploaded_file).split('.')[0]
            doc_summary = TextAnalysis.summarize_doc(filename=file_path,user_input='give summary for this document')
            doc_summary = doc_summary.split('\n')
            
            return render(request, 'document_summarization.html', {'file_summary': doc_summary, 'file': file_name})
    else:
        messages.error(request, "login first")
        return redirect("doc_page")
    return render(request, "document_summarization.html")

@csrf_exempt
def upload_file(request):
    global filepath
    if request.method == 'POST' and request.user.is_authenticated:
   
        if request.FILES.get('file'):
            uploaded_file = request.FILES['file']
                
            filepath = ''
            dir = process_uploaded_file(uploaded_file)
            os.makedirs(dir, exist_ok=True)
            fs = FileSystemStorage(location=dir)
            saved_file = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(dir+"/"+saved_file)
            filepath += file_path
            print(file_path)
            return JsonResponse({'message': 'file uploaded successfully'})
    else:
        messages.error(request, "please login first")
        return redirect("qa")

    return JsonResponse({'message': 'Invalid request'}, status=400)

def process_uploaded_file(uploaded_file):
    try:
        upload_dir = 'E:/react-django/django_sessions/uploads'
        return upload_dir
    except Exception as e:
        return f'Error processing file: {str(e)}'
    
@csrf_exempt
def document_question_answer(request):
    
    global user_input
    if request.method == 'POST':
        try:
            user_input = ''
            data = json.loads(request.body.decode('utf-8'))
            user_data = data.get('user_data', '')
            user_data = cleantext.clean(user_data,extra_spaces=True,lowercase=True)
            user_input += user_data
            print(user_input)
            questions_and_answers = TextAnalysis.question_answer(filename=filepath, user_question=user_input)
            
            response_data = {'questions_and_answers': str(questions_and_answers)}
            
            return JsonResponse(response_data)
        except json.JSONDecodeError as e:
            return JsonResponse({'message': f'Error decoding JSON: {str(e)}'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request method'}, status=405)
