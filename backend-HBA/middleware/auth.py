from jose import jwt, JWTError
from fastapi import HTTPException, Header, Response
from typing import Optional, Dict
from datetime import datetime, timedelta

from config.app_config import get_settings
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class AuthService:
    
    @staticmethod
    def authenticate_token(authorization: Optional[str] = Header(None)) -> Dict[str, str]:
      
        if not authorization:
            logger.warning("Missing authorization header")
            raise HTTPException(
                status_code=401,
                detail="Authorization header required. Please include 'Authorization: Bearer <token>'"
            )
        
        if not authorization.startswith("Bearer "):
            logger.warning(f"Invalid authorization header format")
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header format. Use 'Bearer <token>'"
            )
        
        try:
            token = authorization.split(" ", 1)[1]
            
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            user_id = payload.get("userId") or payload.get("user_id")
            email = payload.get("email")
            role = payload.get("role", "user")
            
            if not user_id or not email:
                logger.error(f"Token missing required fields - userId: {user_id}, email: {email}")
                raise HTTPException(
                    status_code=401,
                    detail="Token missing required fields (userId and email)"
                )
            
            user_data = {
                "userId": str(user_id),
                "email": email,
                "role": role
            }
            
            logger.debug(f"Authentication successful for user: {user_data['email']}")
            return user_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(status_code=401, detail="Token has expired")
        except JWTError as e:
            logger.error(f"Invalid token: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
        except IndexError:
            logger.error("Malformed authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed")
    
    @staticmethod
    def get_current_user_email(authorization: Optional[str] = Header(None)) -> str:
        user_data = AuthService.authenticate_token(authorization)
        return user_data["email"]
    
    @staticmethod
    def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
        user_data = AuthService.authenticate_token(authorization)
        return user_data["userId"]
    
    @staticmethod
    def create_jwt_token(user_data: Dict[str, str]) -> str:
       
        payload = {
            "userId": user_data["userId"],
            "email": user_data["email"],
            "role": user_data.get("role", "user"),
            "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRY_HOURS),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    
    @staticmethod
    def generate_rolling_token_response(response: Response, current_user: Dict[str, str]) -> None:
       
        try:
            required_fields = ["userId", "email"]
            for field in required_fields:
                if field not in current_user:
                    logger.warning(f"Missing {field} in current_user data")
                    return
            
            new_token = AuthService.create_jwt_token(current_user)
            response.headers["x-access-token"] = new_token
            logger.debug(f"Generated rolling token for user: {current_user['email']}")
            
        except Exception as e:
            logger.error(f"Error generating rolling token: {e}")


def get_current_user_email(authorization: Optional[str] = Header(None)) -> str:
   return AuthService.get_current_user_email(authorization)


def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    return AuthService.get_current_user_id(authorization)


def authenticate_token(authorization: Optional[str] = Header(None)) -> Dict[str, str]:
      return AuthService.authenticate_token(authorization)