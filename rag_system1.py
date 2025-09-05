from neo4j import GraphDatabase
from typing import List, Dict, Optional, Tuple
import re

class EnhancedGraphHandler:
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="neo4j123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def get_entity_count(self) -> Dict[str, int]:
        """Get count of nodes and relationships"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN count(n) as node_count
            """)
            node_count = result.single()["node_count"]
            
            result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as rel_count
            """)
            rel_count = result.single()["rel_count"]
            
            return {"nodes": node_count, "relationships": rel_count}
    
    def find_all_entities(self, entity_type: str = None) -> List[str]:
        """Find all entities or entities of specific type"""
        with self.driver.session() as session:
            if entity_type:
                result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.name CONTAINS $type
                    RETURN n.name as name
                    LIMIT 50
                """, type=entity_type)
            else:
                result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.name as name
                    LIMIT 50
                """)
            return [record["name"] for record in result]
    
    def find_relationships_by_type(self, rel_type: str) -> List[Dict]:
        """Find all relationships of a specific type"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Entity)-[r:RELATION {type: $rel_type}]->(b:Entity)
                RETURN a.name as source, r.type as relationship, b.name as target
                LIMIT 20
            """, rel_type=rel_type)
            return [dict(record) for record in result]
    
    def find_path_between_entities(self, entity1: str, entity2: str, max_depth: int = 3) -> List[Dict]:
        """Find path between two entities"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (a:Entity)-[*1..$max_depth]-(b:Entity)
                WHERE a.name CONTAINS $entity1 AND b.name CONTAINS $entity2
                RETURN path
                LIMIT 5
            """, entity1=entity1, entity2=entity2, max_depth=max_depth)
            return [dict(record) for record in result]
    
    def get_entity_connections(self, entity_name: str, depth: int = 2) -> List[Dict]:
        """Get all connections of an entity up to specified depth"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:Entity)-[*1..$depth]-(connected)
                WHERE start.name CONTAINS $entity_name
                RETURN start.name as source, connected.name as target
                LIMIT 30
            """, entity_name=entity_name, depth=depth)
            return [dict(record) for record in result]
    
    def find_most_connected_entities(self, limit: int = 10) -> List[Dict]:
        """Find entities with most connections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)-[r:RELATION]-()
                RETURN n.name as entity, count(r) as connection_count
                ORDER BY connection_count DESC
                LIMIT $limit
            """, limit=limit)
            return [dict(record) for record in result]
    
    def get_relationship_types(self) -> List[str]:
        """Get all unique relationship types"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN DISTINCT r.type as rel_type
                ORDER BY rel_type
            """)
            return [record["rel_type"] for record in result]
    
    def search_entities_by_pattern(self, pattern: str) -> List[str]:
        """Search entities by name pattern"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.name CONTAINS $pattern
                RETURN n.name as name
                LIMIT 20
            """, pattern=pattern)
            return [record["name"] for record in result]
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics"""
        with self.driver.session() as session:
            # Node count
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            
            # Relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            
            # Unique relationship types
            result = session.run("MATCH ()-[r:RELATION]->() RETURN count(DISTINCT r.type) as rel_types")
            rel_types = result.single()["rel_types"]
            
            # Most connected entity
            result = session.run("""
                MATCH (n:Entity)-[r:RELATION]-()
                RETURN n.name as entity, count(r) as connections
                ORDER BY connections DESC
                LIMIT 1
            """)
            most_connected = result.single()
            
            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "unique_relationship_types": rel_types,
                "most_connected_entity": most_connected["entity"] if most_connected else None,
                "max_connections": most_connected["connections"] if most_connected else 0
            }

def test_enhanced_graph():
    """Test the enhanced graph functionality"""
    try:
        graph = EnhancedGraphHandler()
        
        print("🔍 Enhanced Graph Query Test")
        print("=" * 40)
        
        # Test basic statistics
        stats = graph.get_graph_statistics()
        print(f"📊 Graph Statistics:")
        print(f"   Nodes: {stats['total_nodes']}")
        print(f"   Relationships: {stats['total_relationships']}")
        print(f"   Relationship Types: {stats['unique_relationship_types']}")
        print(f"   Most Connected: {stats['most_connected_entity']} ({stats['max_connections']} connections)")
        
        # Test relationship types
        rel_types = graph.get_relationship_types()
        print(f"\n🔗 Relationship Types: {rel_types}")
        
        # Test entity search
        entities = graph.find_all_entities()
        print(f"\n👥 Sample Entities: {entities[:5]}")
        
        print("\n✅ Enhanced graph queries working!")
        
    except Exception as e:
        print(f"❌ Error testing enhanced graph: {e}")

if __name__ == "__main__":
    test_enhanced_graph()
