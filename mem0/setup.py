from setuptools import setup, find_packages

setup(
    name="mem0ai",
    version="0.1.101",
    description="Long-term memory for AI Agents (Local Development Version)",
    packages=find_packages(),
    python_requires=">=3.9,<4.0",
    install_requires=[
        "qdrant-client>=1.9.1",
        "pydantic>=2.7.3",
        "openai>=1.33.0",
        "posthog>=3.5.0",
        "pytz>=2024.1",
        "sqlalchemy>=2.0.31",
    ],
    extras_require={
        "graph": [
            "langchain-neo4j>=0.4.0",
            "neo4j>=5.23.1",
            "rank-bm25>=0.2.2",
        ],
    },
)