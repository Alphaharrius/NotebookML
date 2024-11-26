from pydantic import BaseModel, Field, field_serializer, field_validator
from flatdict import FlatDict
from unflatten import unflatten

from abc import ABC
from typing import Optional, LiteralString, List, Iterable
from datetime import datetime

import hashlib
import chromadb
import chromadb.config


class VectorDatabase:
    """ Database interface. """

    class DBConfig(ABC, BaseModel):
        """ Base class for database configuration. """
        data_container_name: str = Field(..., description="Name of the data container.")
        " Name of the data container. "

    class PersistentConfig(DBConfig):
        """ Configuration for a persistent database. """
        path: str = Field(..., description="Path to the persistent database file.")
        " Path to the persistent database file. "

    @classmethod
    def persistant(cls, config: PersistentConfig):
        """
        Create a persistent database.
        
        Args:
            config (PersistentClient): Configuration for the database.
        """
        api = chromadb.PersistentClient(
            path=config.path,
            settings=chromadb.config.Settings(),
            tenant=chromadb.config.DEFAULT_TENANT,
            database=chromadb.config.DEFAULT_DATABASE)
        api.get_or_create_collection(config.data_container_name)
        return cls(api=api, config=config)
    
    IMPL = {
        PersistentConfig: persistant
    }
    " Mapping from configuration type to database creation method. "

    @classmethod
    def from_config(cls, config: DBConfig):
        """ Create a database from a configuration. """
        return cls.IMPL[type(config)](cls, config)

    def __init__(self, api: chromadb.ClientAPI, config: DBConfig):
        self.api = api
        " Database API. "
        self.config = config
        " Database configuration. "

    def create_collection(self, name: str):
        """ Create a collection. """
        self.api.create_collection(name)
    
    def has_collection(self, name: str):
        """ Check if a collection exists. """
        collection_names = set(v.name for v in self.api.list_collections())
        return name in collection_names
    
    def search(self, collection_name: LiteralString, descriptions: LiteralString, limit: int) -> List['Data']:
        """
        Get data from the database.
        
        Args:
            collection_name (str): Name of the collection to search.
            descriptions (str): Description of the data to search for.
            limit (int): Maximum number of results to return.
        """
        if not self.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        collection = self.api.get_collection(collection_name)
        results = collection.query(query_texts=descriptions, n_results=limit)
        metadatas = results['metadatas'][0]
        contents = results['documents'][0]

        return [Data(meta=DataMeta.model_validate(unflatten(metadata)), content=content) 
                for metadata, content in zip(metadatas, contents)]
    
    def add(self, collection_name: LiteralString, items: Iterable['Data']) -> None:
        """
        Add data to the database.
        
        Args:
            collection_name (str): Name of the collection to add to.
            items (Iterable[Data]): Data to add.
        """
        if not self.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        collection = self.api.get_collection(collection_name)
        collection.add(
            documents=[v.content for v in items],
            metadatas=[dict(FlatDict(v.meta.model_dump(), delimiter='.')) for v in items],
            ids=[meta_identifier(v.meta) for v in items])


class MetaData(ABC, BaseModel):
    """ Metadata for an entry. """
    created_datetime: datetime = Field(..., description="Date and time of the entry.")
    " Date of the entry. "

    @field_serializer('created_datetime')
    def serialize_dt(self, dt: datetime, _info):
        return dt.timestamp()
    
    @field_validator("created_datetime", mode="before")
    @classmethod
    def validate_dt(cls, v, _info) -> datetime:
        if isinstance(v, float): return datetime.fromtimestamp(v)
        return v


class EntityMeta(MetaData):
    """ Metadata for an entity. """
    type: str = Field(..., description="Type of the entity.", enum=["undef", "user", "agent"])
    " Type of the entity. "
    name: Optional[str] = Field('', description="Name of the entity if entity is an agent.")
    " Name of the entity if entity is an agent. "
    id: str = Field(..., description="Unique identifier of the entity.")
    " Unique identifier of the entity. "


NULL_ENTITY = EntityMeta(created_datetime=datetime.strptime("2000-1-1 00:00:00", "%Y-%m-%d %H:%M:%S"), 
                         type="undef", 
                         name="null", 
                         id="")
" Metadata for a null entity. "


class SourceMeta(MetaData):
    """ Metadata for a source. """
    type: str = Field(..., description="Type of the source.", enum=["user", "document", "website"])
    " Type of the source. "
    ref: Optional[str] = Field('', description="Reference of the source for type `document`.")
    " Reference of the source for type `document`. "
    url: Optional[str] = Field('', description="URL of the source for type `website`.")
    " URL of the source for type `website`. "


class DataMeta(MetaData):
    """ Metadata for a database entry. """
    type: str = Field(..., description="Type of the entry.", enum=["data", "content"])
    " Type of the entry. "
    owner: EntityMeta = Field(..., description="Owner of the entry.")
    " Owner of the entry. "
    creator: EntityMeta = Field(..., description="Creator of the entry.")
    " Creator of the entry. "
    source: SourceMeta = Field(..., description="Source of the data.")
    " Source of the data. "


class Data(BaseModel):
    """ Data entry. """
    meta: DataMeta = Field(..., description="Metadata for the data.")
    " Metadata for the data. "
    content: str = Field(..., description="Content of the data.")
    " Content of the data. "


def meta_identifier(meta: MetaData) -> LiteralString:
    """
    Get an unique identifier for the metadata.
    
    Args:
        meta (MetaData): Metadata to get identifier for.
    """
    meta_encoded = meta.model_dump_json().encode("utf-8")
    meta_code = hashlib.sha256(meta_encoded).hexdigest()
    meta_subcode = hashlib.sha1(meta_encoded).hexdigest()
    return f"{meta_code}:{meta_subcode}"
