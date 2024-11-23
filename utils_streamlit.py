import bcrypt

def generate_password_hash(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

def check_password_hash(stored_hash, password):
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash)