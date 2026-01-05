# gym_app.py
import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime, date
from deepface import DeepFace
from PIL import Image
import smtplib
from email.message import EmailMessage
import shutil

# -------------------------
# Config / paths / defaults
# -------------------------
st.set_page_config(page_title="Gym Face Entry System", layout="wide")

MEM_FILE = "members.csv"
ATT_FILE = "attendance.csv"
DELETED_FILE = "deleted_members.csv"
IMG_DIR = "member_images"
CONFIG_FILE = "config.json"

os.makedirs(IMG_DIR, exist_ok=True)

# Recognition model + threshold
RECOGNITION_MODEL = "VGG-Face"
DISTANCE_THRESHOLD = 0.6   # adjust if needed (higher -> more permissive)

# -------------------------
# CSV initialization helpers
# -------------------------
def ensure_csv_files():
    if not os.path.exists(MEM_FILE):
        pd.DataFrame(columns=["ID","Name","Gender","Email","Mobile","Membership","Fee","JoinDate","ImagePath"]).to_csv(MEM_FILE, index=False)
    if not os.path.exists(ATT_FILE):
        pd.DataFrame(columns=["ID","Name","Date","EntryTime","ExitTime","Status"]).to_csv(ATT_FILE, index=False)
    if not os.path.exists(DELETED_FILE):
        pd.DataFrame(columns=["ID","Name","Gender","Email","Mobile","Membership","Fee","JoinDate","ImagePath","DeletedAt"]).to_csv(DELETED_FILE, index=False)

def load_members():
    ensure_csv_files()
    df = pd.read_csv(MEM_FILE, dtype=str).fillna("")
    # keep ID column as string for safe comparisons
    if "ID" not in df.columns:
        df["ID"] = ""
    return df

def save_members(df):
    df.to_csv(MEM_FILE, index=False)

def load_attendance():
    ensure_csv_files()
    df = pd.read_csv(ATT_FILE, dtype=str).fillna("")
    for c in ["ID","Name","Date","EntryTime","ExitTime","Status"]:
        if c not in df.columns:
            df[c] = ""
    return df

def save_attendance(df):
    df.to_csv(ATT_FILE, index=False)
def load_deleted():
    ensure_csv_files()
    try:
        df = pd.read_csv(DELETED_FILE, dtype=str).fillna("")
        if df.empty or df.columns.tolist() == [""]:
            raise Exception("Empty file")
        return df
    except:
        # recreate file with correct columns
        df = pd.DataFrame(columns=[
            "ID","Name","Gender","Email","Mobile",
            "Membership","Fee","JoinDate","ImagePath","DeletedAt"
        ])
        df.to_csv(DELETED_FILE, index=False)
        return df


def save_deleted(df):
    df.to_csv(DELETED_FILE, index=False)

# -------------------------
# Config load/save (SMTP)
# -------------------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

cfg = load_config()

# -------------------------
# Email util
# -------------------------
def send_email_using_config(to_email, subject, body):
    cfg_local = load_config()
    admin_email = cfg_local.get("admin_email")
    admin_pass = cfg_local.get("admin_pass")
    if not admin_email or not admin_pass:
        return False, "SMTP not configured. Set in sidebar."
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = admin_email
        msg["To"] = to_email
        msg.set_content(body)
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(admin_email, admin_pass)
            smtp.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

# -------------------------
# Face verification helper
# -------------------------
def try_verify_faces(img1_path, img2_path):
    """
    Returns (match_bool, distance_or_none, details)
    """
    try:
        res = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name=RECOGNITION_MODEL, enforce_detection=False)
        verified = bool(res.get("verified", False))
        distance = None
        for k in ("distance","cosine","euclidean"):
            if k in res:
                try:
                    distance = float(res[k])
                    break
                except Exception:
                    continue
        if distance is not None:
            match = distance <= DISTANCE_THRESHOLD
        else:
            match = verified
        return match, distance, res
    except Exception as e:
        return False, None, {"error": str(e)}

# -------------------------
# ID generator (unique)
# -------------------------
def generate_member_id():
    members = load_members()
    if members.empty:
        return "1"
    # find numeric IDs and take max + 1
    nums = []
    for v in members["ID"].astype(str).tolist():
        try:
            nums.append(int(v))
        except:
            continue
    if not nums:
        return "1"
    return str(max(nums) + 1)

# -------------------------
# Save image helper
# -------------------------
def save_member_image(uploaded_file, member_id, name):
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    filename = f"{member_id}_{safe_name}.jpg"
    path = os.path.join(IMG_DIR, filename)
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img.save(path, format="JPEG", quality=90)
    except Exception as e:
        # If uploaded_file is bytes
        with open(path, "wb") as f:
            f.write(uploaded_file)
    return path

# -------------------------
# Sidebar: SMTP config & menu
# -------------------------
st.sidebar.title("🏋️ Gym Admin Settings")
with st.sidebar.expander("SMTP Settings (one-time)", expanded=False):
    admin_email = st.text_input("Admin Gmail (sender)", value=cfg.get("admin_email",""))
    admin_pass = st.text_input("Admin App Password (Gmail App Password)", type="password")
    if st.button("Save SMTP Settings"):
        if not admin_email or not admin_pass:
            st.warning("Provide both email and app password.")
        else:
            save_config({"admin_email": admin_email, "admin_pass": admin_pass})
            st.success("SMTP settings saved locally to config.json")

st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Register Member", "Update / Delete Member", "Attendance - Entry", "Attendance - Exit", "View Members", "View Attendance", "Reset DB"])

st.title("Gym Face Entry System")

ensure_csv_files()

# -------------------------
# 1) Register Member
# -------------------------
if menu == "Register Member":
    st.header("Register New Member")
    with st.form("reg_form"):
        id_input = st.text_input("Member ID (optional, leave blank for auto)")
        name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        email = st.text_input("Email")
        mobile = st.text_input("Mobile Number")
        membership = st.selectbox("Membership Type", ["Monthly", "Quarterly", "Yearly"])
        fee_default = {"Monthly":500,"Quarterly":1400,"Yearly":5000}.get(membership,0)
        fee = st.number_input("Fee (₹)", min_value=0, value=fee_default, step=1)
        join_date = st.date_input("Join Date", value=date.today())
        photo = st.camera_input("Capture Face Photo")
        submit = st.form_submit_button("Register Member")

    if submit:
        if not (name and email and mobile and photo):
            st.warning("Please fill Name, Email, Mobile and capture a photo.")
        else:
            members = load_members()
            member_id = id_input.strip() if id_input.strip() else generate_member_id()
            # ensure unique
            if member_id in members["ID"].astype(str).values:
                st.error("Member ID already exists. Choose different ID or leave blank for auto ID.")
            else:
                # save image
                try:
                    img_path = save_member_image(photo, member_id, name)
                except Exception as e:
                    st.error(f"Failed to save image: {e}")
                    img_path = ""
                new_row = {
                    "ID": member_id,
                    "Name": name,
                    "Gender": gender,
                    "Email": email,
                    "Mobile": mobile,
                    "Membership": membership,
                    "Fee": str(fee),
                    "JoinDate": str(join_date),
                    "ImagePath": img_path
                }
                members = pd.concat([members, pd.DataFrame([new_row])], ignore_index=True)
                save_members(members)
                st.success(f"Member registered: {name} (ID: {member_id})")
                # send email if configured
                subject = "Gym Registration Successful"
                body = f"Hello {name},\n\nYour registration is successful.\nMember ID: {member_id}\nMembership: {membership}\nFee: ₹{fee}\nJoin Date: {join_date}\n\nRegards,\nGym Team"
                sent, info = send_email_using_config(email, subject, body)
                if sent:
                    st.info("Registration email sent to member.")
                else:
                    st.warning(f"Registration saved but email not sent: {info}")

# -------------------------
# 2) Update / Delete Member
# -------------------------
elif menu == "Update / Delete Member":
    st.header("Update or Delete Member")
    members = load_members()
    if members.empty:
        st.warning("No members available.")
    else:
        sel_id = st.selectbox("Select Member ID", options=["--select--"] + members["ID"].astype(str).tolist())
        if sel_id and sel_id != "--select--":
            member = members[members["ID"].astype(str) == sel_id].iloc[0]
            st.markdown(f"**Member:** {member['Name']}  •  **Email:** {member['Email']}")
            with st.form("update_form"):
                name = st.text_input("Name", member["Name"])
                gender = st.selectbox("Gender", ["Male","Female","Other"], index=["Male","Female","Other"].index(member["Gender"]) if member["Gender"] in ["Male","Female","Other"] else 0)
                email = st.text_input("Email", member["Email"])
                mobile = st.text_input("Mobile", member["Mobile"])
                membership = st.selectbox("Membership Type", ["Monthly","Quarterly","Yearly"], index=["Monthly","Quarterly","Yearly"].index(member["Membership"]) if member["Membership"] in ["Monthly","Quarterly","Yearly"] else 0)
                fee = st.number_input("Fee (₹)", value=float(member.get("Fee",0)))
                join_date = st.date_input("Join Date", value=pd.to_datetime(member["JoinDate"]).date() if member["JoinDate"] else date.today())
                new_photo = st.camera_input("Capture New Photo (optional)")
                upd = st.form_submit_button("Update Member")
            if upd:
                img_path = member["ImagePath"]
                if new_photo:
                    try:
                        img_path = save_member_image(new_photo, sel_id, name)
                    except Exception as e:
                        st.warning(f"Could not save new photo: {e}")
                members.loc[members["ID"].astype(str) == sel_id, ["Name","Gender","Email","Mobile","Membership","Fee","JoinDate","ImagePath"]] = [name,gender,email,mobile,membership,str(fee),str(join_date),img_path]
                save_members(members)
                st.success("Member updated.")
                # notify
                sub = "Gym Membership Updated"
                bod = f"Hello {name},\n\nYour membership details have been updated.\nMember ID: {sel_id}\nMembership: {membership}\nFee: ₹{fee}\n\nRegards,\nGym Team"
                s, i = send_email_using_config(email, sub, bod)
                if s:
                    st.info("Update email sent.")
                else:
                    st.warning(f"Update saved but email not sent: {i}")

            if st.button("Delete Member (permanent)"):
                # backup row
                row = members[members["ID"].astype(str) == sel_id].iloc[0].to_dict()
                deleted = load_deleted()
                row["DeletedAt"] = str(datetime.now())
                deleted = pd.concat([deleted, pd.DataFrame([row])], ignore_index=True)
                save_deleted(deleted)
                # delete image
                try:
                    if row.get("ImagePath") and os.path.exists(row["ImagePath"]):
                        os.remove(row["ImagePath"])
                except Exception:
                    pass
                # delete member row
                members = members[members["ID"].astype(str) != sel_id]
                save_members(members)
                # delete attendance rows
                attendance = load_attendance()
                attendance = attendance[attendance["ID"].astype(str) != str(sel_id)]
                save_attendance(attendance)
                st.warning(f"Member {sel_id} deleted and backed up.")
                # notify
                sub = "Gym Membership Deleted"
                bod = f"Hello {row.get('Name','')},\n\nYour membership (ID: {sel_id}) has been deleted.\n\nRegards,\nGym Team"
                s, i = send_email_using_config(row.get("Email",""), sub, bod)
                if s:
                    st.info("Deletion email sent.")
                else:
                    st.warning(f"Deletion done but email not sent: {i}")

# -------------------------
# 3) Attendance - Entry
# -------------------------
elif menu == "Attendance - Entry":
    st.header("Attendance — Entry (Face Verification)")
    st.write("Capture live face to mark Entry. System ensures one row per person per day.")
    uploaded = st.camera_input("Capture Face for Entry")

    if uploaded is not None:
        try:
            temp_path = "temp_entry.jpg"
            Image.open(uploaded).convert("RGB").save(temp_path, format="JPEG", quality=90)
        except Exception as e:
            st.error(f"Failed to read camera image: {e}")
            temp_path = None

        if temp_path:
            members = load_members()
            if members.empty:
                st.warning("No registered members to match.")
            else:
                best = None
                best_dist = 1e9
                progress = st.progress(0)
                total = len(members)
                for i, (_, row) in enumerate(members.iterrows(), 1):
                    progress.progress(int(i/total*100))
                    member_img = row.get("ImagePath","")
                    if not member_img or not os.path.exists(member_img):
                        continue
                    match, distance, details = try_verify_faces(temp_path, member_img)
                    if distance is None:
                        # fallback boolean
                        if match:
                            best = (row, distance)
                            break
                    else:
                        if distance < best_dist:
                            best_dist = distance
                            best = (row, distance)
                        if distance <= DISTANCE_THRESHOLD:
                            break
                progress.empty()

                if not best:
                    st.error("No matching member found in database.")
                else:
                    row, dist = best
                    is_match = (dist is not None and dist <= DISTANCE_THRESHOLD) or (dist is None and True)
                    if is_match:
                        attendance = load_attendance()
                        now = datetime.now()
                        today = now.strftime("%Y-%m-%d")
                        idstr = str(row["ID"])
                        exists = attendance[(attendance["ID"].astype(str) == idstr) & (attendance["Date"] == today)]
                        if not exists.empty:
                            st.warning(f"Entry already marked today for {row['Name']}.")
                        else:
                            new_entry = {"ID": idstr, "Name": row["Name"], "Date": today, "EntryTime": now.strftime("%H:%M:%S"), "ExitTime": "", "Status": "Present"}
                            attendance = pd.concat([attendance, pd.DataFrame([new_entry])], ignore_index=True)
                            save_attendance(attendance)
                            st.success(f"Entry recorded for {row['Name']} at {new_entry['EntryTime']}")
                            if dist is not None:
                                st.write(f"Match distance: {dist:.4f}")
                    else:
                        st.error("Face did not match sufficiently. Try again or register.")

# -------------------------
# 4) Attendance - Exit
# -------------------------
elif menu == "Attendance - Exit":
    st.header("Attendance — Exit (Face Verification)")
    st.write("Capture live face to mark Exit. This updates the same row (Entry + Exit).")
    uploaded = st.camera_input("Capture Face for Exit")

    if uploaded is not None:
        try:
            temp_path = "temp_exit.jpg"
            Image.open(uploaded).convert("RGB").save(temp_path, format="JPEG", quality=90)
        except Exception as e:
            st.error(f"Failed to read camera image: {e}")
            temp_path = None

        if temp_path:
            members = load_members()
            if members.empty:
                st.warning("No registered members to match.")
            else:
                best = None
                best_dist = 1e9
                progress = st.progress(0)
                total = len(members)
                for i, (_, row) in enumerate(members.iterrows(), 1):
                    progress.progress(int(i/total*100))
                    member_img = row.get("ImagePath","")
                    if not member_img or not os.path.exists(member_img):
                        continue
                    match, distance, details = try_verify_faces(temp_path, member_img)
                    if distance is None:
                        if match:
                            best = (row, distance)
                            break
                    else:
                        if distance < best_dist:
                            best_dist = distance
                            best = (row, distance)
                        if distance <= DISTANCE_THRESHOLD:
                            break
                progress.empty()

                if not best:
                    st.error("No matching member found in database.")
                else:
                    row, dist = best
                    is_match = (dist is not None and dist <= DISTANCE_THRESHOLD) or (dist is None and True)
                    if is_match:
                        attendance = load_attendance()
                        now = datetime.now()
                        today = now.strftime("%Y-%m-%d")
                        idstr = str(row["ID"])
                        mask = (attendance["ID"].astype(str) == idstr) & (attendance["Date"] == today)
                        open_rows = attendance[mask & ((attendance["ExitTime"].isna()) | (attendance["ExitTime"] == ""))]
                        if open_rows.empty:
                            st.warning("No open entry found for today. Please mark Entry first.")
                        else:
                            idx = open_rows.index[0]
                            attendance.at[idx, "ExitTime"] = now.strftime("%H:%M:%S")
                            attendance.at[idx, "Status"] = "Exited"
                            save_attendance(attendance)
                            st.success(f"Exit recorded for {row['Name']} at {attendance.at[idx,'ExitTime']}")
                            if dist is not None:
                                st.write(f"Match distance: {dist:.4f}")
                    else:
                        st.error("Face did not match sufficiently. Try again or register.")

# -------------------------
# 5) View Members (filters)
# -------------------------
elif menu == "View Members":
    st.header("Members List")
    members = load_members()
    if members.empty:
        st.warning("No members registered.")
    else:
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            id_filter = st.text_input("Filter by ID (exact)")
        with c2:
            name_filter = st.text_input("Search Name (partial)")
        with c3:
            phone_filter = st.text_input("Search Mobile (partial)")

        df = members.copy()
        if id_filter:
            df = df[df["ID"].astype(str) == id_filter]
        if name_filter:
            df = df[df["Name"].str.contains(name_filter, case=False, na=False)]
        if phone_filter:
            df = df[df["Mobile"].str.contains(phone_filter, na=False)]

        if df.empty:
            st.info("No matching members.")
        else:
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Members CSV", data=csv, file_name="members_filtered.csv", mime="text/csv")

# -------------------------
# 6) View Attendance (filters)
# -------------------------
elif menu == "View Attendance":
    st.header("Attendance Log")
    attendance = load_attendance()
    if attendance.empty:
        st.warning("No attendance records.")
    else:
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            date_filter = st.date_input("Filter by Date", value=None)
            # To allow none, if user doesn't set date we treat as all
        with c2:
            id_filter = st.text_input("Filter by ID")
        with c3:
            name_filter = st.text_input("Filter by Name")

        df = attendance.copy()
        if date_filter:
            df = df[df["Date"] == str(date_filter)]
        if id_filter:
            df = df[df["ID"].astype(str) == id_filter]
        if name_filter:
            df = df[df["Name"].str.contains(name_filter, case=False, na=False)]

        if df.empty:
            st.info("No matching attendance records.")
        else:
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Attendance CSV", data=csv, file_name="attendance_filtered.csv", mime="text/csv")

# -------------------------
# 7) Reset DB
# -------------------------
elif menu == "Reset DB":
    st.header("Reset Database (Dangerous)")
    st.warning("This will permanently delete members, attendance, deleted members, images, and config.json")
    if st.button("Delete All Data (IRREVERSIBLE)"):
        try:
            for f in [MEM_FILE, ATT_FILE, DELETED_FILE, CONFIG_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(IMG_DIR):
                shutil.rmtree(IMG_DIR)
            os.makedirs(IMG_DIR, exist_ok=True)
            ensure_csv_files()
            st.success("All data removed and fresh CSVs created.")
        except Exception as e:
            st.error(f"Failed to reset DB: {e}") 
