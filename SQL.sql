CREATE TABLE Info (
    infoId VARCHAR(50) PRIMARY KEY,
    departureDate DATE,
    arrivalDate DATE,
    departure VARCHAR(100),
    destination VARCHAR(100)
);

CREATE TABLE Passenger (
    passengerID VARCHAR(50) PRIMARY KEY,
    lastName VARCHAR(50),
    firstName VARCHAR(50),
    gender VARCHAR(10),
    birthDate DATE,
    nationality VARCHAR(50)
);

-- 3. 訂單 (Order)
CREATE TABLE `Order` (
    orderID VARCHAR(50) PRIMARY KEY,
    orderPrice DECIMAL(10, 2),
    orderDate DATE
);

-- 4. 交易 (Transaction)
-- 包含與訂單的 1:1 關係 (產生 Generates)
CREATE TABLE `Transaction` (
    transNo VARCHAR(50) PRIMARY KEY,
    cardID VARCHAR(50),
    cardName VARCHAR(100),
    expiryDate VARCHAR(10),
    securityCode VARCHAR(10),
    transTime DATETIME,
    orderID VARCHAR(50) UNIQUE, -- 1:1 關係的外鍵
    FOREIGN KEY (orderID) REFERENCES `Order`(orderID)
);

-- 5. 航班 (Itinerary)
CREATE TABLE Itinerary (
    ItineraryNum VARCHAR(50) PRIMARY KEY,
    departureTime DATETIME,
    arrivalTime DATETIME,
    travelTime INT
);

-- 6. 航空公司 (Airline)
CREATE TABLE Airline (
    airlineCode VARCHAR(50) PRIMARY KEY,
    airlineName VARCHAR(100)
);

-- 7. 航段 (flightSegment)
-- 包含與航空公司的 N:1 關係 (經營 Operates)
CREATE TABLE flightSegment (
    segmentID VARCHAR(50) PRIMARY KEY,
    departureTime DATETIME,
    arrivalTime DATETIME,
    airportName VARCHAR(100),
    duration INT,
    airlineCode VARCHAR(50), -- N:1 關係的外鍵
    FOREIGN KEY (airlineCode) REFERENCES Airline(airlineCode)
);

-- 8. 機票 (Ticket)
-- 匯集了多個關聯的外鍵與關係屬性
CREATE TABLE Ticket (
    ticketId VARCHAR(50) PRIMARY KEY,
    cabinClass VARCHAR(50),
    
    -- [關係] 包含 Contains (訂單 1 : N 機票)
    orderID VARCHAR(50),
    ticketNum INT,                   -- 關係屬性：機票數量
    ticketTotalPrice DECIMAL(10, 2), -- 關係屬性：機票總價
    
    -- [關係] 對應 Corresponds (旅程資訊 1 : N 機票)
    infoId VARCHAR(50),
    
    -- [關係] 屬於 Belongs (乘客 1 : 1 機票)
    passengerID VARCHAR(50) UNIQUE,
    
    -- [關係] 對應 Corresponds (機票 N : 1 航班)
    ItineraryNum VARCHAR(50),
    
    FOREIGN KEY (orderID) REFERENCES `Order`(orderID),
    FOREIGN KEY (infoId) REFERENCES Info(infoId),
    FOREIGN KEY (passengerID) REFERENCES Passenger(passengerID),
    FOREIGN KEY (ItineraryNum) REFERENCES Itinerary(ItineraryNum)
);

-- 9. 航班_航段 關聯表 (Itinerary_flightSegment)
-- 處理 航班與航段之間的 N:M 關係 (包含 Includes)
CREATE TABLE Itinerary_flightSegment (
    ItineraryNum VARCHAR(50),
    segmentID VARCHAR(50),
    PRIMARY KEY (ItineraryNum, segmentID),
    FOREIGN KEY (ItineraryNum) REFERENCES Itinerary(ItineraryNum),
    FOREIGN KEY (segmentID) REFERENCES flightSegment(segmentID)
);