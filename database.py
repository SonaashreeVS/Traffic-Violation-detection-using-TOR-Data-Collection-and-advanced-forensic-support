# database_fix.py - SQLite version (no MySQL needed)
import sqlite3
import os
from datetime import datetime

class SQLiteViolationDB:
    def __init__(self, db_path='traffic_violations.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # License plates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL UNIQUE,
                total_violations INTEGER DEFAULT 0,
                red_light_violations INTEGER DEFAULT 0,
                helmet_violations INTEGER DEFAULT 0,
                speeding_violations INTEGER DEFAULT 0,
                other_violations INTEGER DEFAULT 0,
                first_violation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_violation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Traffic violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                violation_type TEXT,
                vehicle_type TEXT,
                detected_speed INTEGER DEFAULT 0,
                speed_limit INTEGER DEFAULT 0,
                confidence REAL,
                location_lat REAL,
                location_lng REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                evidence_data TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON traffic_violations(plate_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON traffic_violations(violation_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON traffic_violations(timestamp)')
        
        conn.commit()
        conn.close()
        print("✅ SQLite database initialized successfully!")
    
    def add_violation(self, violation_data):
        """Add a violation to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            plate_number = violation_data.get('license_plate', 'UNKNOWN')
            violation_type = violation_data.get('type', 'UNKNOWN')
            vehicle_type = violation_data.get('vehicle_type', 'Unknown')
            confidence = violation_data.get('confidence', 0.0)
            
            # Update license_plates table
            cursor.execute('''
                INSERT OR REPLACE INTO license_plates 
                (plate_number, total_violations, red_light_violations, helmet_violations, speeding_violations, last_violation)
                VALUES (?, 
                    COALESCE((SELECT total_violations FROM license_plates WHERE plate_number = ?), 0) + 1,
                    COALESCE((SELECT red_light_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    COALESCE((SELECT helmet_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    COALESCE((SELECT speeding_violations FROM license_plates WHERE plate_number = ?), 0) + ?,
                    CURRENT_TIMESTAMP
                )
            ''', (
                plate_number, plate_number, plate_number,
                1 if violation_type == 'RED_LIGHT_VIOLATION' else 0,
                plate_number,
                1 if violation_type == 'NO_HELMET' else 0,
                plate_number,
                1 if violation_type == 'SPEEDING' else 0
            ))
            
            # Insert into traffic_violations table
            cursor.execute('''
                INSERT INTO traffic_violations 
                (plate_number, violation_type, vehicle_type, detected_speed, speed_limit, confidence, evidence_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                plate_number,
                violation_type,
                vehicle_type,
                violation_data.get('detected_speed', 0),
                violation_data.get('speed_limit', 0),
                confidence,
                str(violation_data)
            ))
            
            conn.commit()
            print(f"✅ Violation recorded in SQLite: {plate_number} - {violation_type}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
        finally:
            conn.close()
    
    def get_statistics(self):
        """Get violation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(DISTINCT plate_number) as unique_offenders,
                    SUM(red_light_violations) as total_red_light,
                    SUM(helmet_violations) as total_helmet,
                    SUM(speeding_violations) as total_speeding
                FROM license_plates
            ''')
            stats = cursor.fetchone()
            
            # Top offenders
            cursor.execute('''
                SELECT plate_number, total_violations 
                FROM license_plates 
                ORDER BY total_violations DESC 
                LIMIT 10
            ''')
            top_offenders = cursor.fetchall()
            
            return {
                'total_violations': stats[0] if stats else 0,
                'unique_offenders': stats[1] if stats else 0,
                'violation_breakdown': {
                    'red_light': stats[2] if stats else 0,
                    'no_helmet': stats[3] if stats else 0,
                    'speeding': stats[4] if stats else 0
                },
                'top_offenders': [{'plate': row[0], 'count': row[1]} for row in top_offenders]
            }
            
        except Exception as e:
            print(f"❌ Statistics error: {e}")
            return {}
        finally:
            conn.close()

# Test the database
if __name__ == "__main__":
    db = SQLiteViolationDB()
    print("✅ SQLite database is ready!")