from jose import jwt  
import os
from fastapi import HTTPException, Header, Response  
from typing import Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Environment variables with validation
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
        logger.error("JWT_SECRET not found in environment variables")
        raise ValueError("JWT_SECRET environment variable is required")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "1"))

logger.info(f"JWT Configuration loaded - Algorithm: {JWT_ALGORITHM}, Expiry: {JWT_EXPIRY_HOURS}h")

def authenticate_token(authorization: Optional[str] = Header(None)):
    
    logger.debug(f"Received authorization header: {authorization}")
    
    if not authorization:
        logger.error("No authorization header provided")
        raise HTTPException(
            status_code=401, 
            detail="Authorization header required. Please include 'Authorization: Bearer <token>'"
        )
    
    try:
        # Extract token from "Bearer <token>"
        if not authorization.startswith("Bearer "):
            logger.error(f"Invalid auth header format: {authorization}")
            raise HTTPException(
                status_code=401, 
                detail="Invalid authorization header format. Use 'Bearer <token>'"
            )
        
        token = authorization.split(" ", 1)[1]  
        logger.debug(f"Extracted token: {token[:20]}...")
        
        # Verify token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        logger.debug(f"Token payload: {payload}")
        
        # Handle both Node.js and FastAPI token formats
        user_id = payload.get("userId") or payload.get("user_id")
        email = payload.get("email")
        role = payload.get("role", "user")
        
        # Validate required fields
        if not user_id or not email:
            logger.error(f"Token missing required fields - userId: {user_id}, email: {email}")
            raise HTTPException(
                status_code=401, 
                detail="Token missing required fields (userId/user_id and email)"
            )
            
        user_data = {
            "userId": user_id,
            "email": email,
            "role": role
        }
        
        logger.debug(f"Authentication successful for user: {user_data['email']}")
        return user_data
        
    except jwt.ExpiredSignatureError:
        logger.error("Token expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except IndexError:
        logger.error("Malformed authorization header")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")

def get_current_user_email(authorization: Optional[str] = Header(None)) -> str:
    user_data = authenticate_token(authorization)
    return user_data["email"]

def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    user_data = authenticate_token(authorization)
    return user_data["userId"]

def generate_rolling_token_response(response: Response, current_user: dict) -> None:
    try:
     
        required_fields = ["userId", "email"]
        for field in required_fields:
            if field not in current_user:
                logger.warning(f"Missing {field} in current_user data")
                return
            
        payload = {
            "userId": current_user["userId"],
            "email": current_user["email"],
            "role": current_user.get("role", "user"),
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow() 
        }
            
        new_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        response.headers["x-access-token"] = new_token
        logger.debug(f"Generated rolling token for user: {current_user['email']}")
        
    except Exception as e:
        logger.error(f"Error generating rolling token: {e}")

def create_jwt_token(user_data: dict) -> str:
    payload = {
        "userId": user_data["userId"],
        "email": user_data["email"],
        "role": user_data.get("role", "user"),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow()
    }
        
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)