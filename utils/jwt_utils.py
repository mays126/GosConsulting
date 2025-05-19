import jwt
from pathlib import Path
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Union, Dict

http_bearer = HTTPBearer()

BASE_DIR = Path(__file__).parent.parent
PUBLIC_KEY_PATH = BASE_DIR / "certs" / "public.pem"
PRIVATE_KEY_PATH = BASE_DIR / "certs" / "private.pem"


def encode_jwt(
        payload: Dict,
        private_key: str = PRIVATE_KEY_PATH.read_text(),
        algorithm: str = "RS256"
) -> str:
    """
    Encodes a JWT using a private key.
    """
    return jwt.encode(payload, private_key, algorithm=algorithm)


def decode_jwt(
        token: Union[str, bytes],
        public_key: str = PUBLIC_KEY_PATH.read_text(),
        algorithm: str = "RS256"
) -> Union[Dict, None]:
    """
    Decodes a JWT token using the public key.
    """
    try:
        return jwt.decode(token, public_key, algorithms=[algorithm])
    except Exception as e:
        print(e)
        return None


def hash_password(password: str) -> bytes:
    """
    Hashes a plaintext password.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt)


def validate_password(password: str, hashed: bytes) -> bool:
    """
    Validates a plaintext password against a hashed password.
    """
    return bcrypt.checkpw(password.encode(), hashed)


def get_current_user(
        creds: HTTPAuthorizationCredentials = Depends(http_bearer)
) -> Dict:
    token = creds.credentials
    payload = decode_jwt(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return payload
