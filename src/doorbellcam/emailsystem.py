import smtplib


email = "om.guin@gmail.com"

reciever_email = "pranav.sambhu@gmail.com"




subject = "[URGENT] GUN DETECTED ON CAMPUS"
message = 'A FIREARM HAS BEEN DETECTED BY A CCTV CAMERA ON CAMPUS. THIS IS NOT A TEST. LOCK DOWN IMMEDIATELY'

text = f"Subject: {subject}\n\n{message}"
port = 0 #change this
server = smtplib.SMTP("smtp.gmail.com", port)

server.starttls()
google_app_pass = "blooporium" # change this
server.login(email, google_app_pass)

server.sendmail(email, reciever_email, text)
print("Email sent")