import json
import os
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def create_email_from_json_file(json_file_path, attachment_folder, output_folder):
    """
    Reads email content from a JSON file and generates emails based on the data.

    Args:
        json_file_path (str): Path to the JSON file containing email details.
        attachment_folder (str): Path to the folder containing attachment files.
        output_folder (str): Path to the folder where .eml files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load email data from the JSON file
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        # Assume the file contains a list of email objects
        email_data_list = json.load(json_file)

    for email_data in email_data_list:
        # Unpack email data
        email_id = email_data["email_id"]
        sender = email_data["sender"]
        sender_email = email_data["sender_email"]
        subject = email_data["subject"]
        body = email_data["body"]
        attachments = email_data.get("attachments", [])
        received_at = email_data["received_at"]
        is_duplicate = email_data["is_duplicate"]

        # Create email
        def generate_email(output_filename, is_duplicate=False):
            msg = MIMEMultipart()
            msg['From'] = f"{sender} <{sender_email}>"
            msg['To'] = "support@company.com"
            msg['Subject'] = subject
            msg['Date'] = datetime.strptime(
                received_at, "%Y-%m-%dT%H:%M:%S.%f").strftime("%a, %d %b %Y %H:%M:%S")

            # Add email body
            msg.attach(MIMEText(body, 'plain'))

            # Attach files if available
            for attachment_name in attachments:
                attachment_path = os.path.join(
                    attachment_folder, attachment_name)
                if os.path.exists(attachment_path):
                    with open(attachment_path, "rb") as attachment_file:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment_file.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename={attachment_name}",
                        )
                        msg.attach(part)
                else:
                    print(f"Attachment not found: {attachment_path}")

            # Save the email as .eml
            email_path = os.path.join(output_folder, output_filename)
            with open(email_path, "w", encoding="utf-8") as eml_file:
                eml_file.write(msg.as_string())

            print(f"Generated email: {email_path}")

        # Generate primary email
        generate_email(f"{email_id}.eml")

        # Generate duplicate email if needed
        if is_duplicate:
            duplicate_email_id = f"{email_id}_DUPLICATE"
            generate_email(f"{duplicate_email_id}.eml", is_duplicate=True)


# Example Usage:
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
json_file_path = os.path.join(RESOURCES_DIR, 'email_datasets.json')
attachment_folder_path = os.path.join(
    RESOURCES_DIR, 'attachments')  # Folder containing attachments
# Folder to save generated emails
output_folder_path = os.path.join(RESOURCES_DIR, 'emails')

create_email_from_json_file(
    json_file_path, attachment_folder_path, output_folder_path)
