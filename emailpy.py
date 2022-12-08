import requests


api_key = "77586ac6-2be2-40ee-acff-99f4244dfaf2" 
flag = False
def validate(email_address):
    print(email_address)
    response = requests.get(
        "https://isitarealemail.com/api/email/validate",
        params = {'email': email_address},
        headers = {'Authorization': "Bearer " + api_key })

    status = response.json()['status']
    print(status)
    if status == "valid":
        print(email_address+" : email is valid")
        return False
    elif status == "invalid":
        print(email_address+" : email is invalid")
        return True
    else:
        print(email_address+" : email was unknown")
        return True



def run(firstname, lastname, domin):
    for i in range(1, 10):

        email = firstname +'iln606'+'@'+ domin
        flag = validate(email)
        if flag == False:
           return 'Email is untrackable'

        email = firstname +'@'+ domin
        flag = validate(email)
        if flag == False:
           return email

        email = firstname + lastname + '@' + domin
        flag = validate(email)
        if flag == False:
            return email

        email = lastname + firstname + '@' + domin
        flag = validate(email)
        if flag == False:
            return email

        email = lastname +'@'+ domin
        flag = validate(email)
        if flag == False:
            return email


        email = firstname + '.' + lastname + '@' + domin
        flag = validate(email)
        if flag == False:
           return email

        email = lastname + '.' + firstname + '@' + domin
        flag = validate(email)
        if flag == False:
            return email
        email = firstname + '_' + lastname + '@' + domin
        flag = validate(email)
        if flag == False:
            return email
        email = lastname + '_' + firstname + '@' + domin
        flag = validate(email)
        if flag == False:
            return email
        if i == 11:
            return "No email found"
if __name__ == '__main__':
   
    firstname = 'pradeep12'
    lastname = 'odela'
    domin = 'gmail.com'
    email = run(firstname, lastname, domin)
    print(email+' -----')