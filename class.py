from pinecone import Pinecone
import openai
import warnings
from dotenv import load_dotenv
import os
import re
from datetime import datetime
import logging

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeRetriever:
    HTTP_METHODS = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'}
    
    def __init__(self):
        warnings.filterwarnings("ignore")
        
        # Ortam değişkenlerini yükle
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not self.pinecone_api_key or not openai.api_key:
            raise ValueError("API anahtarları eksik!")

        # Pinecone'u başlat
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index('web-log-datas')

    def _extract_http_filters(self, query):
        """HTTP method ve status code'ları sorguda filtrele"""
        filters = {}
        clean_query = query.upper()

        # HTTP Method filtreleme
        for method in self.HTTP_METHODS:
            if method in clean_query:
                filters["http_method"] = method
                clean_query = clean_query.replace(method, "")

        # Status Code filtreleme
        status_code_pattern = r'\b([1-5][0-9]{2})\b'
        status_codes = re.findall(status_code_pattern, query)
        if status_codes:
            filters["status_code"] = {"$in": [int(code) for code in status_codes]}
            clean_query = re.sub(status_code_pattern, "", clean_query)
        
        clean_query = clean_query.lower().strip()
        
        return filters, clean_query

    def retrieve(self, query, top_k=20, namespace="changeembed_namespace", emb_model="text-embedding-3-small"):
        """Pinecone'dan dokümanları getir"""
        try:
            # HTTP filtreleri ve semantic sorguyu ayır
            filters, semantic_query = self._extract_http_filters(query)
            logger.info(f"Filtreler: {filters}")
            logger.info(f"Semantic sorgu: {semantic_query}")

            # Embedding oluştur
            try:
                query_response = openai.Embedding.create(
                    input=semantic_query,
                    model=emb_model
                )
                query_emb = query_response['data'][0]['embedding']
                logger.info("Embedding başarıyla oluşturuldu")
            except Exception as e:
                logger.error(f"Embedding oluşturma hatası: {e}")
                return [], []

            # Pinecone sorgusu
            try:
                query_result = self.index.query(
                    vector=query_emb,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True,
                    filter=filters if filters else None
                )
                logger.info(f"Pinecone sorgu sonucu: {query_result}")
            except Exception as e:
                logger.error(f"Pinecone sorgu hatası: {e}")
                return [], []

            # Sonuç kontrolü
            if not query_result.get("matches"):
                logger.warning("Sonuç bulunamadı")
                # Filtresiz tekrar dene
                try:
                    logger.info("Filtresiz sorgu deneniyor...")
                    query_result = self.index.query(
                        vector=query_emb,
                        top_k=top_k,
                        namespace=namespace,
                        include_metadata=True
                    )
                    logger.info(f"Filtresiz sorgu sonucu: {query_result}")
                except Exception as e:
                    logger.error(f"Filtresiz sorgu hatası: {e}")
                    return [], []

            # Sonuçları işle
            retrieved_docs = []
            sources = []
            
            for result in query_result.get("matches", []):
                metadata = result["metadata"]
                
                summary = metadata.get("summary", "")
                doc_date = metadata.get("date_time")
                url = metadata.get("url", "")
                
                # HTTP bilgilerini ekle
                http_info = f"[{metadata.get('http_method', 'N/A')} - {metadata.get('status_code', 'N/A')}] "
                summary_with_info = http_info + summary
                
                retrieved_docs.append(summary_with_info)
                sources.append((doc_date, url))

            return retrieved_docs, sources

        except Exception as e:
            logger.error(f"Genel hata: {e}")
            return [], []

def main():
    try:
        retriever = PineconeRetriever()
        query = "how many 500 code with delete"
        
        # Önce index'in mevcut olup olmadığını kontrol et
        try:
            stats = retriever.index.describe_index_stats()
            logger.info(f"Index istatistikleri: {stats}")
        except Exception as e:
            logger.error(f"Index istatistikleri hatası: {e}")
        
        documents, sources = retriever.retrieve(
            query=query,
            top_k=20,
            namespace="changeembed_namespace",
            emb_model="text-embedding-3-small"
        )
        
        print("Documents:", documents)
        print("Sources:", sources)
    
    except Exception as e:
        logger.error(f"Main fonksiyon hatası: {e}")

if __name__ == "__main__":
    main()