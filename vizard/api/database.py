
__all__ = [
    'create_engine', 'declarative_base', 'Session'
    'Base', 'VizardCountryNames'
]

# core
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
# helper
import logging

# configure logger
logger = logging.getLogger(__name__)


# create a DeclarativeMeta instance
Base = declarative_base()


class VizardCountryNames(Base):  # type: ignore
    """Country names used in our application. See :mod:`vizard.data.preprocessor`.
    """
    # DATABASE CONFIG
    __tablename__ = 'country'
    index = Column(Integer, primary_key=True)
    name = Column(String(25))


class VizardMaritalStatusNames(Base):  # type: ignore
    """Marital status names in our application. See :mod:`vizard.data.preprocessor`.
    """
    # DATABASE CONFIG
    __tablename__ = 'marital_status'
    index = Column(Integer, primary_key=True)
    name = Column(String(25))
