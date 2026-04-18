#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m' 

echo -e "${GREEN}🚀 Starting Banking Intent Classification System...${NC}"


python3 scripts/inference.py

echo -e "\n${GREEN}👋 System closed.${NC}"